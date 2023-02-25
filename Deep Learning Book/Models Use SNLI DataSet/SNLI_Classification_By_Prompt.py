import torch
from openprompt.data_utils import InputExample, FewShotSampler
from openprompt.plms import get_model_class, load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, MixedTemplate
from openprompt import PromptForClassification, PromptDataLoader, ClassificationRunner
from openprompt.data_utils.text_classification_dataset import PROCESSORS
import sys
sys.path.append('../Models Use IMDB DataSet')
import Read_IMDB_DataSet

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]

train_dataset, train_label, test_dataset, test_label = Read_IMDB_DataSet.get_train_test_dataset()

dataset = [InputExample(guid=index, text_a=text, label=label) for index, (text, label) in enumerate(zip(train_dataset, train_label))]

# dataset = [ # For simplicity, there's only two examples
#     # text_a is the input text of the SecondQuestionData, some other datasets may have multiple input sentences in one example.
#     InputExample(
#         guid=0,
#         text_a="Albert Einstein was one of the greatest intellects of his time.",
#     ),
#     InputExample(
#         guid=1,
#         text_a="The film was badly made.",
#     ),
# ]


model_class = get_model_class('bert')    # 定义模型的类型
model_path = '../../../Model/bert-base-uncase'

bertConfig = model_class.config.from_pretrained(model_path)
bertTokenizer = model_class.tokenizer.from_pretrained(model_path)    # 就是获取tokenizer
bertModel = model_class.model.from_pretrained(model_path)    # 获取Bert模型
bertWrapper = model_class.wrapper


promptTemplate = ManualTemplate(    # 定义模板
    text='{"placeholder": "text_a"} It was {"mask"}!',
    tokenizer=bertTokenizer
)

promptVerbalizer = ManualVerbalizer(    # 定义一个标签的映射关系 把标签映射到词汇表
    classes=classes,
    label_words={
        "negative": ["bad", "terrible", "horrible"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer=bertTokenizer
)

promptModel = PromptForClassification(    # 定义一个提示学习的模型
    template=promptTemplate,
    verbalizer=promptVerbalizer,
    plm=bertModel
)

promptDataloader = PromptDataLoader(
    dataset=dataset,
    tokenizer=bertTokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=bertWrapper,
)

# promptModel.train()
# trainer = ClassificationRunner(model=promptModel, loss_function="cross_entropy",
#                                train_dataloader=promptDataloader)



promptModel.eval()    # 实现零样本的情感分类
with torch.no_grad():
    for batch in promptDataloader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])

