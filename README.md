# VectorDB

Vector DB standalone service based on Chroma DB.  Including DB manage frontend.


## 起因

在 [Intelligence Integration System](https://github.com/SleepySoft/IntelligenceIntegrationSystem/tree/dev) 项目中，
我并没有直接使用Chroma DB的原生服务，而是通过嵌入使用，并定制接口。

由于我希望在不影响主服务的前提下由另一个脚本建立全数据索引，因此我将Chroma DB做成服务，并加入一个前端页面用于数据管理。

与Chroma DB原生服务相比，本服务支持以下额外功能：

+ 文本分块

+ 分块文本与文档索引（UUID）关联

+ 聚类分析

+ 网页形式的管理页面

# 文件



# 部署


# 常见问题与排查 (Troubleshooting)

### 1. 导入包时报错：`ImportError: DLL load failed... 应用程序控制策略已阻止此文件。`

**问题描述：**
在使用 `pip` 安装带有 C 语言编译扩展（如 `hdbscan`, `scikit-learn`, `numpy` 等）的 Python 包时，由于 Windows 内置的安全机制（网络标记、Windows Defender 或智能应用控制），下载的 `.pyd` 文件可能被拦截，导致无法 `import`。

**解决方案（按推荐顺序）：**

**方案 A：一键解除环境目录下的网络锁定（最常用、最快）**
Windows 默认会拦截带有“从网络下载”标记的脚本或动态链接库。以管理员身份运行 PowerShell，并执行以下命令批量解除锁定（请替换为你的实际 Conda 环境路径）：
```powershell
Get-ChildItem -Path "C:\你的\Conda\环境路径\" -Recurse | Unblock-File

```

**方案 B：优先使用 Conda 安装**
相比 `pip`，Conda 提供的二进制包签名方式不同，通常能直接绕过此类 Windows 安全策略限制。如果 `pip` 安装报错，请先卸载再用 `conda` 安装：

```bash
pip uninstall <package_name>
conda install -c conda-forge <package_name>

```

**方案 C：将环境目录加入 Windows 安全中心白名单**
如果上述方法无效，可能是杀毒软件在实时拦截：

1. 打开 **Windows 安全中心** -> **病毒和威胁防护** -> **管理设置**。
2. 找到 **排除项**，点击 **添加或删除排除项**。
3. 将你的 Conda 虚拟环境整个文件夹加入白名单。

**方案 D：关闭“智能应用控制”（仅限 Windows 11）**
Windows 11 的“智能应用控制 (Smart App Control)”会无差别拦截无签名的开发者库。如果你是开发者，建议在系统设置中搜索并将其直接**关闭**。

```
