# sentiment-classification-base-on-dgcnn-and-attention
文件前的序号为技术迭代路线
dgcnn和attention-pooling来自苏剑林https://spaces.ac.cn/
attention-concatenate是自创的，算一个小小的创新
整体结构采用四通道，主要为了解决显存的局限，使得模型可以在训练的时候用完整的句子，不必为了显存不够而减少句子的长度
