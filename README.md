# pinyin2hanzi
拼音转汉字,  convert pinyin to 汉字 using deep networks


##  TRAINING DATA
As a light-weight example, training data are downloaded from the AI shell speech recognition corpus, 
found in http://openslr.org/33/. The transcripts rather than the audio data are used. A copy of the transcript file is found in the ./data folder

## MODEL ARCHITECTURE

![](./doc/model.png)

## Requirements

1. The project uses pytorch (>=1.3.0) and torchtext. Python3.7 is recommended




## TO DO

1. git clone https://github.com/ranchlai/pinyin2hanzi.git
2. cd pinyin2hanzi
3. virtualenv -p python3.7 py37
4. source py37/bin/activate
5. pip install -r requirements.txt
6. Run process_ai_shell_transcript_sd.py to convert ai-shell transcripts from 汉字 to 带声调的拼音(pinyin with tones)
7. Run train.py to train the model, or you can download the pretrained model and put it to the "model" subfolder
7. Run inference_sd.py to do inference


##  Pretrained model

Pretrained model using AI-shell transcript file can be downloaded from 
[gooole drive](https://drive.google.com/open?id=186jnywHwnxqXDBxrbFRpIF7dFAWcwEx_)




##  一些带声调拼音的测试结果如下, results obtained by running inference_sd.py


néng gòu yíng de bǐ sài zhēn de hěn kāi xīn

能够赢得比赛真的很开心

yě qǔ de le sān xiàn piāo hóng de chéng jī

也取得了三线飘红的成绩

guó yǒu qǐ yè bù bì yě wú xū jiè rù yíng lì xìng qiáng de shāng pǐn fáng kāi fā

国有企业不必也无需介入盈利性强的商品房开发

sī fǎ jiàn dìng jī gòu shì dú lì fǎ rén

司法鉴定机构是独立法人

bǎo bǎo zhòng wǔ diǎn èr jīn

宝宝重五点二斤

## Reference


Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.

