0. 数据文件[地址](https://snap.stanford.edu/data/feather-lastfm-social.html)
1. create_graph.py 文件的作用是将lasftm_asia文件下的内容整理成graph.pkl文件，
节点有以下几个属性：
    ①feature：由lastfm_asia_features.json生成，是以one-hot编码形式保存，每个节点的feature长度等于json中出现的不同数字的个数（详情看create_graph.py 第40 -48行）
    ②label：由lastfm_asia_target.csv生成
    ③degree：保存当前节点的度
2. graph.bin 文件是图的dgl版
3. 五个json存储了图的相关特征
4. [analysis](https://github.com/fduer/Social-Computing/tree/main/analysis), [classification](https://github.com/fduer/Social-Computing/tree/main/classification), [link prediction](https://github.com/fduer/Social-Computing/tree/main/link_prediction)和[community detection](https://github.com/fduer/Social-Computing/tree/main/community_detection)分别存放了相关任务结果和代码。
