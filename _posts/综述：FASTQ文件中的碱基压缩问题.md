---
title:  有关于FASTQ文件中的碱基序列压缩问题的笔记
date: 2021-08-09 21:21:08
tags: 生物信息学
---

目前 FASTQ 文件的无损压缩算法主要分为四类。第一类即使用普适的压缩软件 对文件进行压缩存储，例如 Gzip 以及 Bzip2，该类方法将 FASTQ 文件看做普通的字 符输入，不对其做任何处理，因此无法充分利用文件潜在的数据特征，使得压缩效果 不够理想。

第二类算法的主要思想是重新对文件中序列进行排序，使得相似的序列尽可能聚 集从而提高压缩率，例如 SCALCE，Orcom，Mince以及 BEETL等，其中 BEETL 的核心思想为针对序列集合的轻量级的 BWT 变换，该变换实现对序 列集合的顺序重置以尽可能使得序列集满足 RLO 排序，从而大幅提升压缩效果。 除此之外，还可以使用聚类思想重置序列顺序以增大相似序列的聚集度。算法通过判 断序列间的汉明距离将相似的序列进行重新分类，并使用适用于高度重复序列的压缩 器对同一类序列进行压缩处理，例如 FQC使用哈希算法将具有相似片段的 DNA 序 列映射至同一类，为每一类选择一条参考序列，并计算集合中其他序列与之差异，使 用概率预测模型对差异信息进行压缩存储；该算法对质量分数同样采取聚类思想，首 先生成多条参考序列并根据汉明距离将所有质量分数序列进行分类并压缩存储差异 信息。LCTD 算法使用聚类算法对 DNA 序列分类并重置顺序以便充分利用序列 间的冗余以及序列相似性，LCTD 算法对于质量分数的压缩思想与 FQC 类似，同样 根据数据特征产生多条序列作为聚类中心，并据此进行序列分类，算法最终使用 ZPAQ 算法作为压缩器。该类算法对高度重复的序列集合有十分理想的压缩效果， 但对于 FASTQ 文件中的序列进行重新排序，可能改变测序结果所表达的生物意义， 且重新排序后的标识符序列冗余性将大大降低，并需要记录序列位置信息以便恢复原 始文件，因此可能影响压缩率以及压缩时间。

第三类 FASTQ 文件无损压缩算法则使用参考序列，使用模糊匹配以及精确匹配 技术将序列与参考序列进行对比并存储差异，该类方法的优势在于当序列长度增加时， 相比于无参考序列的压缩算法，该类算法将取得更优的压缩率。Quip是该类算法的代表，该算法可以根据输入文件自动生成较优的参考序列而不需要借助其他数据库。除此之外，Quip 算法的另一压缩模式为马尔科夫预测模型，算法选择一阶或高阶马 尔科夫链对序列集合进行压缩存储。Leon和 KIC也同样采用了类似的压缩策略， 但需要注意的是，序列映射以及重组均需要密集的计算，因此上述算法可能会牺牲高 效性以实现更为理想的压缩率。

第四类算法首先对 FASTQ 文件进行预处理，并使用统计建模以及熵编码对预处 理后的数据进行压缩存储。例如 G-SQZ将 DNA 和质量分数序列的对应字符进行组 合，形成<核苷酸，质量分数>二元组作为一个新的符号，使用哈夫曼编码对二元组 符号进行压缩；DSRC 则通过将输入分块，结合 LZ77 编码，游程编码以及哈夫曼 编码来进行高效压缩。Fqzcomp 和 Fastqz 算法对标识符间的差异使用差量编码， 使用算数编码以及文本模型对 DNA 序列以及质量分数序列进行处理，Fastqz 算法 同时支持基于参考序列的 DNA 序列压缩；KungFQ首先对序列进行分组编码，并 使用 Gzip 作为压缩器，同时该算法支持有损压缩，对质量分数序列进行一定程度的 平滑从而提高压缩率；LFQC存储一条标识符序列范式，并据此对所有标识符序列 进行分区存储，对不同类型区域使用相应处理方式，使用 ZPAQ 算法压缩 DNA 序列、 质量分数序列以及经过预处理后的标识符序列。该算法可以达到较为优异的压缩效果， 但是运行速度较慢。综上所述，FASTQ 文件的有损压缩算法更注重下游基因分型测 试的准确性，因此允许数据产生一定精度的损失。而无损压缩算法以保证数据完整性 为前提，大部分算法采取不同方式对输入文件进行预处理，以现有的成熟压缩算法作 为压缩器，从而尽可能地提升文件压缩效果。

 

#### LFQC

We have used the same algorithm as the quality scores for sequence compression. The FASTQ sequences are strings of five possible characters namely A;C;G;T;N. If the sequence contains ‘N’s for unknown nucleotides, these usually all have the lowest quality score (which is encoded by ‘!’ with ASCII code 33). Therefore, we could remove the ‘N’ characters from the sequences and replace their quality scores by a character not used for other quality scores. However, we empirically determined that this did not improve the results by much and it increases the running time, so we did not make use of this observation. Some datasets have color space encoded reads (SOLiD). A color space read starts with a nucleotide (A;C;G;T) followed by numbers 0–3. The numbers encode nucleotides depending on their relationship to the previous nucleotide. For example, 0 means the nucleotide is identical to the previous one, and 3 means the nucleotide is the complement of the previous one, etc. We remove all the end of line characters from the sequences and keep the ends of line with the quality scores, since the length of the quality score sequence is the same as the length of the read. Next we apply the lpaq8 compression algorithm with parameter ‘9’ meaning highest compression. The algorithm runs single threaded.

我们使用了与序列压缩质量评分相同的算法。FASTQ序列是由5个可能的字符组成的字符串，即A、C、G、T、N。如果序列包含' N '代表未知的核苷酸，这些通常都有最低的质量分数(编码为' !'， ASCII码是33)。因此，我们可以从序列中删除' N '字符，并用一个不用于其他质量分数的字符替换它们的质量分数。然而，我们根据经验确定，这样做并没有大大提高结果，反而增加了运行时间，所以我们没有利用这一观察结果。一些数据集有彩色空间编码读取(SOLiD)。颜色空间读取以核苷酸(A;C;G;T)开头，后面跟着数字0-3。这些数字根据核苷酸与前一个核苷酸的关系来编码核苷酸。例如，0表示与前一个核苷酸相同，3表示与前一个核苷酸互补，等等。我们从序列中删除所有的行尾字符，并保留带有质量分数的行尾，因为质量分数序列的长度与读取的长度相同。接下来我们应用lpaq8压缩算法，参数' 9 '表示最高压缩。该算法运行单线程。

 

#### GTZ

(1)Read in streams of large data files. (2)Pre-process the input by dividing data streams into three sub-streams: metadata, base sequence, and quality score. (3)Buffer sub-streams in local memories and assemble them into different types of data blocks with a fixed size. (4)Compress assembled data blocks and their descriptions, and then transmit output blocks into the cloud storage.

(1)读入大数据文件流。(2)对输入进行预处理，将数据流分为三个子流：元数据、基序、质量分数。(3)在本地存储器中缓存子流，并将其组装成固定大小的不同类型的数据块。(4)将组装好的数据块及其描述进行压缩，然后将输出块传输到云存储中。

 

The simplest implementation of adaptive modeling is order-0. Exactly, it does not consider any context information, thus this short-sighted modeling can only see the current character and make prediction that is independent of the previous sequences. Similarly, an order-1 encoder makes prediction based on one preceding character. Consequently, the low-order modeling makes little contribution to the performance of compressors. Its main advantage is that it is very memory efficient. Hence, for quality score streams that do not have spatial locality, a low-order modeling is adequate for moderate compression rate. 

 Our tailored low-order encoder for reads is demonstrated in Fig. 5. The first step is to transform sequences with the BWT algorithm. BWT (Burrows-Wheeler transform) rearranges reads into runs of similar characters. In the second step, the zero-order and the first-order prediction model are used to calculate appearance probability of each character. Since a poor probability accuracy contributes to undesirable encoding results, we add interpolation after quantizing the weighted average probability, to reduce prediction errors and improve compression ratios. In the last procedure, the bit arithmetic coding algorithm produces decimals ranging from zero to one as outputs to represent sequences.

自适应建模的最简单实现是阶数为0。准确地说，它没有考虑任何上下文信息，因此这种短视的建模只能看到当前的字符，并做出独立于先前序列的预测。类似地，1阶编码器基于前一个字符进行预测。因此，低阶建模对压气机性能的影响很小。它的主要优点是内存效率非常高。因此，对于没有空间局部性的质量分值流，低阶建模对于中等压缩率是足够的。

图5展示了我们为读取量身定做的低阶编码器。第一步是使用BWT算法转换序列。BWT(Burrow-Wheeler Transform)将读数重新排列为相似字符的串。在第二步中，利用零阶和一阶预测模型计算每个字符的出现概率。由于较差的概率精度会导致不期望的编码结果，因此我们在量化加权平均概率之后添加内插，以减少预测误差并提高压缩比。在最后一个过程中，位算术编码算法产生从0到1的十进制作为表示序列的输出。

 

#### LFastqC

Sequence compression 

The nucleotide sequences are arranged in a small string of five alphabetic characters, namely A, C, G, T, and N. The N base contains unknown nucleotides and always has “!” as its corresponding quality score, which indicates the lowest probability and is equal to zero. Some FASTQ algorithms eliminate “N” in the record sequences or “!” in the record quality score because they can be easily reconstructed from one another. Our algorithm does not follow this approach as we simply use the quality score as a read-length reference. 

Some other datasets use color space encoding, which means that the read sequence has more than five characters. The color-space read sequence starts with any of A, C, G, or T, followed by numbers 0–3, which represent the relationship between the current base and the previous one. Our algorithm supports these datasets because it uses MFCompress, a FASTA and multi-FASTA special-purpose compressor that accepts FASTA files with more than five characters. To compress the record sequences, our algorithm first converts the stream into a single FASTA file by adding the header of the first sequence as the first line, then deleting all sequence reads’ new lines to get a long single sequence read. LFastqC then feeds the converted stream to MFCompress for compression. We use MFCompress with a parameter of -3 and obtain the best compression ratio.

序列压缩

核苷酸序列排列在一个由五个字母字符组成的小字符串中，即A、C、G、T和N。N碱基包含未知核苷酸，并且总是有“！”作为其对应的质量分数，其表示最低概率并且等于零。一些FASTQ算法消除了记录序列中的“N”或“！”这是因为它们可以很容易地从另一个记录质量分数中重建。我们的算法不遵循这种方法，因为我们只使用质量分数作为读取长度参考。

其他一些数据集使用颜色空间编码，这意味着读取序列具有五个以上的字符。颜色空间读取序列以A、C、G或T中的任何一个开始，后跟数字0-3，表示当前基准与前一个基准之间的关系。我们的算法支持这些数据集，因为它使用了MFCompress，这是一种FASTA和多FASTA专用压缩器，可以接受超过五个字符的FASTA文件。为了压缩记录序列，我们的算法首先通过添加第一个序列的头作为第一行，然后删除所有序列读取的新行来将流转换为单个FASTA文件，从而获得长的单个序列读取。然后，LFastqC将转换后的流馈送给MFCompress进行压缩。我们使用参数为-3的MFCompress，获得了最佳的压缩比。
