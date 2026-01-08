import os
import time
import uuid
import logging
import argparse
import tempfile
from enum import Enum
from typing import Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Blueprint, request, jsonify, send_file, Response

# Import the core engine defined in the previous step
try:
    from VectorStorageEngine import VectorStorageEngine
except ImportError:
    from .VectorStorageEngine import VectorStorageEngine


# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# DEFAULT_MODEL = 'BAAI/bge-m3'
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class ServiceUnavailable(Exception):
    class Code(str, Enum):
        INIT = "INIT"
        BUSY = "BUSY"

    def __init__(self, code: str, reason: str):
        self.code = code
        self.reason = reason


class VectorDBService:
    """
    VectorDBService: The Web API Layer.
    
    Responsibilities:
    1. Wraps the VectorStorageEngine with a REST API.
    2. Serves the VectorDBFrontend.html UI.
    3. Can run standalone or mount onto an existing Flask app.
    """

    def __init__(self, engine: VectorStorageEngine, frontend_filename: str = "VectorDBFrontend.html"):
        """
        Args:
            engine (VectorStorageEngine): The initialized storage engine instance.
            frontend_filename (str): The HTML file name located in the same directory.
        """
        self.engine = engine
        self.frontend_filename = frontend_filename
        self._is_registered = False
        
        # Locate the frontend file relative to this script
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self._frontend_path = os.path.join(self._base_dir, self.frontend_filename)

        self._analysis_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AnalysisWorker")
        self._analysis_jobs: Dict[str, Dict[str, Any]] = {}

        self.metrics = {
            'service_unavailable_count': 0,
            'init_errors': 0,
            'busy_errors': 0,
            'request_count': 0
        }

        if not os.path.exists(self._frontend_path):
            logger.warning(f"Frontend file not found at: {self._frontend_path}")

    def create_blueprint(self, wrapper: Optional[Callable] = None) -> Blueprint:
        """
        Creates the Flask Blueprint containing all API routes.

        Args:
            wrapper (Callable): Optional decorator to wrap all routes (e.g., for auth).
        """
        bp = Blueprint("vector_db", __name__, static_folder=None)

        # Helper to apply wrapper if it exists
        def route(rule, **options):
            def decorator(f):
                endpoint = options.pop("endpoint", None)
                if wrapper:
                    f = wrapper(f)
                bp.add_url_rule(rule, endpoint, f, **options)
                return f

            return decorator

        @bp.errorhandler(ServiceUnavailable)
        def handle_service_unavailable(e):
            self.metrics['service_unavailable_count'] += 1

            if e.code == ServiceUnavailable.Code.INIT:
                self.metrics['init_errors'] += 1
            elif e.code == ServiceUnavailable.Code.BUSY:
                self.metrics['busy_errors'] += 1

            if os.environ.get('FLASK_ENV') == 'production':
                return jsonify({
                    "error": "Service temporarily unavailable",
                    "error_code": e.code,
                    "retry_after": 30,
                    "status": "unavailable"
                }), 503
            else:
                # 开发环境可以显示详细错误
                return jsonify({
                    "error": e.reason,
                    "error_code": e.code,
                    "retry_after": 5,
                    "status": "unavailable"
                }), 503

        # --- Helper Method ---

        def get_repo_strict(name: str):
            """
            Retrieves a repository strictly.
            Raises ServiceUnavailable if engine is loading.
            Raises ValueError if collection does not exist.
            """
            # 1. Check readiness
            if not self.engine.is_ready():
                status = self.engine.get_status()
                if status["status"] == VectorStorageEngine.Status.INIT:
                    # 503 Service Unavailable if just loading
                    raise ServiceUnavailable(ServiceUnavailable.Code.INIT, "Engine is initializing")
                elif status["status"] == VectorStorageEngine.Status.ERROR:
                    # 500 Internal Server Error if init failed permanently
                    raise Exception(f"Engine failed to start: {status['error']}")
                else:
                    raise Exception(f"Engine got exception: {status['error']}")

            # 2. Retrieve Repo
            repo = self.engine.get_repository(name)
            if not repo:
                # Raise exception to be caught and returned as 404
                raise ValueError(f"Collection '{name}' not found. Please create it via POST /api/collections first.")
            return repo

        # --- Routes ---

        @route("/")
        def serve_ui():
            """Serves the Single Page Application."""
            if os.path.exists(self._frontend_path):
                return send_file(self._frontend_path)
            return "Frontend HTML not found.", 404

        @route("/api/status")
        def server_status():
            """Returns the engine initialization status."""
            return jsonify(self.engine.get_status())

        @route("/api/health")
        def health_check():
            return jsonify({"status": "ok", "service": "VectorDBService"})

        @route("/api/collections", methods=["POST"])
        def create_collection():
            """
            Creates a new collection or updates config of existing one.
            Body: { "name": str, "chunk_size": int, "chunk_overlap": int }
            """
            data = request.json or {}
            name = data.get("name")
            if not name:
                return jsonify({"error": "Collection name is required"}), 400

            chunk_size = int(data.get("chunk_size", 512))
            chunk_overlap = int(data.get("chunk_overlap", 50))

            try:
                # Use ensure_repository: creates if missing, updates config if exists
                repo = self.engine.ensure_repository(name, chunk_size, chunk_overlap)
                return jsonify({
                    "status": "success",
                    "message": f"Collection '{name}' ready.",
                    "config": repo.get_config()
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/upsert", methods=["POST"])
        def upsert_document(name):
            """
            Modified to use Async Queue.
            Returns 202 Accepted immediately.
            """
            try:
                data = request.json
                doc_id = data.get('doc_id')
                text = data.get('text')
                metadata = data.get('metadata', {})

                # Validation
                if not doc_id or not text:
                    return jsonify({"error": "doc_id and text are required"}), 400

                get_repo_strict(name)   # Call this function to check the engine and repository status in a unified way.

                if self.engine.submit_upsert(name, doc_id, text, metadata):
                    return jsonify({
                        "status": "queued",
                        "message": "Document accepted for processing.",
                        "doc_id": doc_id
                    }), 202
                else:
                    raise ServiceUnavailable(ServiceUnavailable.Code.BUSY, "Server busy (queue full)")

            except ServiceUnavailable as e:
                raise e

            except Exception as e:
                logger.error(f"Upsert request failed: {e}")
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/upsert_batch", methods=["POST"])
        def upsert_batch(name):
            try:
                data = request.json  # Expecting list of {doc_id, text, metadata}
                if not isinstance(data, list):
                    return jsonify({"error": "Expected a list"}), 400

                # 构建内部任务格式
                tasks = []
                for item in data:
                    tasks.append({
                        "collection_name": name,
                        "doc_id": item.get("doc_id"),
                        "text": item.get("text"),
                        "metadata": item.get("metadata", {})
                    })

                get_repo_strict(name)   # Call this function to check the engine and repository status in a unified way.

                if self.engine.submit_upsert_batch(tasks):
                    return jsonify({"status": "queued", "count": len(tasks)}), 202
                else:
                    raise ServiceUnavailable(ServiceUnavailable.Code.BUSY, "Server busy (queue full)")

            except ServiceUnavailable as e:
                raise e

            except Exception as e:
                logger.error(f"Upsert batch request failed: {e}")
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/documents/<doc_id>/exists", methods=["GET"])
        def check_document_exists(name, doc_id):
            try:
                repo = get_repo_strict(name)

                # Use the repo's exists method for single document check
                exists = repo.exists(doc_id)

                return jsonify({
                    "collection": name,
                    "doc_id": doc_id,
                    "exists": exists
                })

            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e
            except Exception as e:
                logger.error(f"Single document existence check failed: {e}")
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/exists", methods=["POST"])
        def check_documents_exist(name):
            try:
                data = request.json or {}
                doc_ids = data.get("doc_ids", [])

                # Validate input
                if not doc_ids:
                    return jsonify({
                        "error": "doc_ids field is required and cannot be empty"
                    }), 400

                if not isinstance(doc_ids, list):
                    return jsonify({
                        "error": "doc_ids must be a list of document IDs"
                    }), 400

                # Check for non-string values in the list
                invalid_ids = [doc_id for doc_id in doc_ids if not isinstance(doc_id, str)]
                if invalid_ids:
                    return jsonify({
                        "error": f"All document IDs must be strings. Invalid IDs: {invalid_ids}"
                    }), 400

                repo = get_repo_strict(name)

                # Use the repo's exists_batch method for efficient batch checking
                # Ensure the repo has this method (should be implemented in VectorCollectionRepo)
                exists_map = repo.exists_batch(doc_ids)

                return jsonify({
                    "collection": name,
                    "exists_map": exists_map,
                    "total_checked": len(doc_ids),
                    "total_exists": sum(1 for exists in exists_map.values() if exists)
                })

            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e
            except Exception as e:
                logger.error(f"Batch document existence check failed: {e}")
                return jsonify({"error": str(e)}), 500

        @route("/api/status/queue", methods=["GET"])
        def queue_status():
            """New endpoint to monitor queue depth."""
            return jsonify(self.engine.get_queue_status())

        @route("/api/collections/<name>/stats", methods=["GET"])
        def get_stats(name):
            try:
                repo = get_repo_strict(name)
                return jsonify({"collection": name, "chunk_count": repo.count()})
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e     # Handled by errorhandler
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/search", methods=["POST"])
        def search(name):
            """
            Expects JSON: { "query": str, "top_n": int, "score_threshold": float, "filter_criteria": dict }
            """
            data = request.json or {}
            query = data.get("query")
            if not query:
                return jsonify({"error": "query string is required"}), 400

            top_n = data.get("top_n", 5)
            score_threshold = data.get("score_threshold", 0.0)
            filter_criteria = data.get("filter_criteria", None)

            try:
                repo = get_repo_strict(name)
                results = repo.search(
                    query_text=query,
                    top_n=top_n,
                    score_threshold=score_threshold,
                    filter_criteria=filter_criteria
                )
                return jsonify(results)
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e     # Handled by errorhandler
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/documents/<doc_id>", methods=["DELETE"])
        def delete_document(name, doc_id):
            try:
                repo = get_repo_strict(name)
                success = repo.delete_document(doc_id)
                if success:
                    return jsonify({"status": "success", "doc_id": doc_id})
                else:
                    return jsonify({"status": "warning", "message": "Document not found"}), 404
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e     # Handled by errorhandler
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/clear", methods=["POST"])
        def clear_collection(name):
            try:
                repo = get_repo_strict(name)
                repo.clear()
                return jsonify({"status": "cleared", "collection": name})
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e     # Handled by errorhandler
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/admin/backup", methods=["GET"])
        def download_backup():
            """
            Triggers a backup and returns the zip file.
            """
            try:
                # Create a temporary directory to store the zip
                temp_dir = tempfile.mkdtemp()
                zip_path = self.engine.create_backup(temp_dir)

                # Send file and perform cleanup afterwards if possible
                filename = os.path.basename(zip_path)
                return send_file(
                    zip_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype='application/zip'
                )
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/admin/restore", methods=["POST"])
        def upload_restore():
            """
            Accepts a zip file upload and restores the database.
            Restarting the service is recommended but Hot Reload is attempted.
            """
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if not file.filename.endswith('.zip'):
                return jsonify({"error": "Only .zip files are allowed"}), 400

            try:
                # Save uploaded file to temp
                temp_fd, temp_path = tempfile.mkstemp(suffix=".zip")
                os.close(temp_fd)
                file.save(temp_path)

                # Trigger engine restore
                self.engine.restore_backup(temp_path)

                # Cleanup upload
                os.remove(temp_path)

                return jsonify({"status": "success", "message": "Database restored and reloaded."})
            except Exception as e:
                return jsonify({"error": f"Restore failed: {str(e)}"}), 500

        # Get list of all collections
        @route("/api/collections", methods=["GET"])
        def list_all_collections():
            try:
                names = self.engine.list_collections()
                return jsonify({"collections": names})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Paginated document browsing
        @route("/api/collections/<name>/documents", methods=["GET"])
        def list_documents(name):
            limit = int(request.args.get("limit", 20))
            offset = int(request.args.get("offset", 0))

            try:
                repo = get_repo_strict(name)
                data = repo.list_documents(limit=limit, offset=offset)
                return jsonify(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except ServiceUnavailable as e:
                raise e     # Handled by errorhandler
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @route("/api/collections/<name>/analysis", methods=["POST"])
        def trigger_analysis(name):
            """
            Submit a clustering analysis job with different clustering methods.

            This endpoint accepts different clustering methods (kmeans, auto_kmeans, dbscan)
            each with their own parameters. The analysis runs asynchronously in the background.
            """
            # Clean up old jobs (older than 1 hour)
            current_time = time.time()
            expired_jobs = [jid for jid, j in self._analysis_jobs.items()
                            if current_time - j['created_at'] > 3600]
            for jid in expired_jobs:
                del self._analysis_jobs[jid]

            # Check if collection exists
            get_repo_strict(name)

            # Parse request data
            data = request.json or {}
            method = data.get("method", "kmeans").lower()  # Default to kmeans
            params = data.get("params", {})

            # Validate method
            valid_methods = ["kmeans", "auto_kmeans", "dbscan"]
            if method not in valid_methods:
                return jsonify({
                    "error": f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}"
                }), 400

            # Validate and prepare parameters based on method
            validated_params = {}

            if method == "kmeans":
                # Required parameters
                n_clusters = params.get("n_clusters")
                if n_clusters is None:
                    return jsonify({
                        "error": "Parameter 'n_clusters' is required for kmeans method"
                    }), 400
                if not isinstance(n_clusters, int) or n_clusters < 2:
                    return jsonify({
                        "error": "Parameter 'n_clusters' must be an integer >= 2"
                    }), 400

                validated_params["n_clusters"] = n_clusters
                validated_params["max_samples"] = params.get("max_samples", 20000)
                validated_params["config"] = {
                    "weights": params.get("weights", {"semantic": 1.0, "time": 0.2, "entities": 0.5})
                }
                validated_params["time_field"] = params.get("time_field", "timestamp")
                validated_params["includes_metas"] = params.get("includes_metas")

            elif method == "auto_kmeans":
                # Validate parameters
                config = params.get("config", {})
                selector_method = config.get("method", "elbow")
                max_k = config.get("max_k", 20)

                if selector_method not in ["elbow", "silhouette"]:
                    return jsonify({
                        "error": "Parameter 'method' in config must be 'elbow' or 'silhouette'"
                    }), 400

                if not isinstance(max_k, int) or max_k < 2:
                    return jsonify({
                        "error": "Parameter 'max_k' must be an integer >= 2"
                    }), 400

                validated_params["max_samples"] = params.get("max_samples", 20000)
                validated_params["config"] = {
                    "method": selector_method,
                    "max_k": max_k,
                    "weights": params.get("weights", {"semantic": 1.0, "time": 0.2, "entities": 0.5})
                }
                validated_params["time_field"] = params.get("time_field", "timestamp")
                validated_params["includes_metas"] = params.get("includes_metas")

            elif method == "dbscan":
                # Validate parameters
                eps = params.get("eps", 0.5)
                if not isinstance(eps, (int, float)) or eps <= 0:
                    return jsonify({
                        "error": "Parameter 'eps' must be a positive number"
                    }), 400

                min_samples = params.get("min_samples", 5)
                if not isinstance(min_samples, int) or min_samples < 1:
                    return jsonify({
                        "error": "Parameter 'min_samples' must be an integer >= 1"
                    }), 400

                validated_params["eps"] = eps
                validated_params["min_samples"] = min_samples
                validated_params["max_samples"] = params.get("max_samples", 20000)
                validated_params["config"] = {
                    "weights": params.get("weights", {"semantic": 1.0, "time": 0.2, "entities": 0.5})
                }
                validated_params["time_field"] = params.get("time_field", "timestamp")
                validated_params["includes_metas"] = params.get("includes_metas")

            # Validate common parameters
            max_samples = validated_params.get("max_samples")
            if not isinstance(max_samples, int) or max_samples <= 0:
                return jsonify({
                    "error": "Parameter 'max_samples' must be a positive integer"
                }), 400

            # Create job ID
            job_id = str(uuid.uuid4())
            self._analysis_jobs[job_id] = {
                "job_id": job_id,
                "collection": name,
                "method": method,
                "params": validated_params,
                "status": "pending",
                "created_at": time.time(),
                "result": None
            }

            # Submit to thread pool
            self._analysis_executor.submit(
                self._run_analysis_task, job_id, name, method, validated_params
            )

            return jsonify({
                "status": "accepted",
                "job_id": job_id,
                "method": method,
                "message": f"{method} analysis started in background."
            }), 202

        @route("/api/analysis/<job_id>", methods=["GET"])
        def get_analysis_result(job_id):
            """
            Step 2: 轮询任务结果
            """
            job = self._analysis_jobs.get(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404

            status = job["status"]

            if status == "completed":
                # 返回结果并清理内存 (Optional: 也可以保留一段时间，这里选择读后即焚或保留)
                # 为了简单，我们不删除，让前端可以多次读取。
                # 生产环境需要定期清理 self._analysis_jobs
                return jsonify({
                    "status": "completed",
                    "data": job["result"]
                })

            elif status == "failed":
                return jsonify({
                    "status": "failed",
                    "error": job.get("error")
                }), 500

            else:
                # pending 或 processing
                return jsonify({
                    "status": status,
                    "message": "Calculation in progress..."
                }), 200

        return bp

    def mount_to_app(self, app: Flask, url_prefix: str = "/vector-db", wrapper: Optional[Callable] = None) -> bool:
        """
        Mount the dashboard to an existing Flask app instance.
        
        Args:
            app (Flask): The main application instance.
            url_prefix (str): Base URL for the dashboard (e.g. /vector-db).
            wrapper (Callable): Optional auth/logging wrapper for routes.
        """
        if self._is_registered:
            logger.warning("VectorDB blueprint already registered.")
            return False

        bp = self.create_blueprint(wrapper)
        app.register_blueprint(bp, url_prefix=url_prefix)
        self._is_registered = True
        logger.info(f"VectorDB Service mounted at {url_prefix}")
        return True

    def run_standalone(self, host="0.0.0.0", port=8001, debug=False):
        """Run as a standalone Flask app."""
        app = Flask(__name__)
        
        # Mount at root for standalone usage
        self.mount_to_app(app, url_prefix="")
        
        print(f"Starting standalone VectorDB at http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)

    def _run_analysis_task(self, job_id: str, collection_name: str, method: str, params: dict):
        """
        Execute analysis task in background thread with specified clustering method.

        This is an internal method that handles the actual clustering execution
        based on the selected method and parameters.
        """
        try:
            # Update status
            self._analysis_jobs[job_id]["status"] = "processing"

            # Get repository
            repo = self.engine.get_repository(collection_name)
            if not repo:
                raise ValueError(f"Collection {collection_name} not found")

            # Execute clustering based on method
            if method == "kmeans":
                result = repo.analyze_clusters(**params)
            elif method == "auto_kmeans":
                result = repo.analyze_clusters_auto_k(**params)
            elif method == "dbscan":
                result = repo.analyze_clusters_dbscan(**params)
            else:
                raise ValueError(f"Unknown clustering method: {method}")

            self._analysis_jobs[job_id]["result"] = result
            self._analysis_jobs[job_id]["status"] = "completed"

        except Exception as e:
            logger.error(f"Analysis job {job_id} failed: {e}")
            self._analysis_jobs[job_id]["error"] = str(e)
            self._analysis_jobs[job_id]["status"] = "failed"


# --- Usage Examples ---


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configuration Logic ---
    parser = argparse.ArgumentParser(description="VectorDB Standalone Service")

    # 1. Network Config
    parser.add_argument("--host", type=str,
                        default=os.getenv("VECTOR_HOST", "0.0.0.0"),
                        help="Host to bind (default: 0.0.0.0 or env VECTOR_HOST)")

    parser.add_argument("--port", type=int,
                        default=int(os.getenv("VECTOR_PORT", 8001)),
                        help="Port to bind (default: 8001 or env VECTOR_PORT)")

    # 2. Storage Config
    parser.add_argument("--db-path", type=str,
                        default=os.getenv("VECTOR_DB_PATH", "./chroma_data"),
                        help="Path to save vector data (default: ./chroma_data or env VECTOR_DB_PATH)")

    # 3. Model Config
    parser.add_argument("--model", type=str,
                        default=os.getenv("VECTOR_MODEL", DEFAULT_MODEL),
                        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL} or env VECTOR_MODEL)")

    args = parser.parse_args()

    print("=" * 50)
    print(f"Starting VectorDB Service")
    print(f" - Host:      {args.host}:{args.port}")
    print(f" - DB Path:   {args.db_path}")
    print(f" - Model:     {args.model}")
    print("=" * 50)

    # --- Initialization ---

    # 1. Initialize Engine with Config
    # Note: ensure directory exists or let Chroma create it
    os.makedirs(args.db_path, exist_ok=True)

    engine_instance = VectorStorageEngine(
        db_path=args.db_path,
        model_name=args.model
    )

    # 2. Initialize Service
    service = VectorDBService(engine=engine_instance)

    # 3. Run Standalone
    service.run_standalone(host=args.host, port=args.port, debug=False)
