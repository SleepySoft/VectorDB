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


