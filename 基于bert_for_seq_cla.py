from datasets import load_dataset,ClassLabel
from pathlib import Path
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score



#  数据路径
Raw_data_dir = Path(__file__).parent / "2.资料" / "2.数据集" / "1.评论数据集"

# 1. 数据处理
def process():
    print('开始处理数据')
    # 读取文件
    dataset = load_dataset("csv", data_files=str(Raw_data_dir/"online_shopping_10_cats.csv"))["train"] # 单数据集直接用train取值
    
    # 过滤数据
    dataset = dataset.remove_columns(["cat"])
    dataset = dataset.filter(lambda x: x["review"] is not None)

    # 划分数据集
    # 强转文本标签为类别标签
    dataset = dataset.cast_column('label', ClassLabel(names=['negative','positive']))
    print(dataset.features)

     # 成功划分训练集和测试集
    dataset_dict = dataset.train_test_split(test_size=0.2,stratify_by_column="label")
    print(dataset_dict)

    # 加载分词器Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    # 构建分词函数
    def batch_encode(batch):
    # example: {label:[1,0,1,0], review: ["**","**", "**","**"]}
        inputs = tokenizer(batch["review"], padding="max_length", max_length=128,truncation=True)
        inputs['labels'] = batch['label']
        return inputs
    
    # 应用分词 删除不需要的字段
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label','review'])

    # 保存数据集
    # arrow格式 官方推荐 适合 dataset and dataset_dict
    dataset_dict.save_to_disk('./arrow_dataset')





# 2. 创建 Dataloader
def get_dataloader(train=True):
    dataset_dict = load_from_disk("./arrow_dataset")
    dataset = dataset_dict["train" if train else "test"]
    dataset.set_format(type='torch') # 指定为pytorch张量
    dataloader = DataLoader(dataset, batch_size=64 , shuffle=True)
    return dataloader



# 4. 训练一次的逻辑
def train_one_epoch(model, dataloader, optimizer, device):
    total_loss = 0
    model.train()
    for batch in tqdm(dataloader,desc='训练'):
        inputs = {k: v.to(device) for k,v in batch.items()}
        
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)



# 5. 训练    
def train():
    #1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    #2. 数据
    dataloader = get_dataloader()

    #4. 模型
    model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese').to(device)

    #6. 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5) # 参数越多, 学习率越小
    

    # 7.训练循环
    best_loss = float('inf')
    for epoch in range(1,21):
        print(f"============Epoch{epoch}==========")
        loss = train_one_epoch(model, dataloader,  optimizer, device)
        print(f'Loss:{loss:.4f}')

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            model.save_pretrained("bert_sentiment_model")
            print('模型已保存')

# 6.预测
def predict(texts):
    if isinstance(texts, str):  #  自动兼容单句字符串输入
        texts = [texts]
    # 1. 确定设备
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    # 3. 加载模型
    model = AutoModelForSequenceClassification.from_pretrained("bert_sentiment_model").to(device)
    print('模型加载成功')

    # 4. 将texts变成张量
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. 批量预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

    # 输出结果
    for text, pred, prob in zip(texts, preds, probs):
        label = "Positive" if pred.item() == 1 else "Negative"
        print(f"\n文本: {text}\n置信度: {prob.item():.4f}\n情感预测: {label}")

# 7.评估
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    dataloader = get_dataloader(train=False)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained("bert_sentiment_model").to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop('labels').cpu().numpy()

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n测试集准确率: {acc:.4f}")



if __name__ == '__main__':

    process()
    train()
    evaluate()
    predict('写的不错,下次别写了')