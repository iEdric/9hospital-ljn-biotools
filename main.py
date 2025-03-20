import os
import pandas as pd
import torch
import base64
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from openai import OpenAI
import pandas as pd
import os,json,re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-fnhyqdrjusu", base_url="https://api.siliconflow.cn/v1")
# Initialize the model
model = Qwen2VLForConditionalGeneration.from_pretrained("olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("olmOCR-7B-0225-preview")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
lock = Lock()  # 创建一个锁对象

def generate_chat_completion(inputStr, model='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'):
    cankaoStr = {
        "就诊医院": "",
        "就诊日期": "",
        "就诊科室": "",
        "姓名": "",
        "性别": "",
        "年龄": "",
        "现病史": "",
        "主诉": "",
        "现病史": "",
        "查体": "",
        "辅助检查": "",
        "诊断": "",
        "处理": "",
        "签名": ""
        }
    message1 = [
            {"role": "system", "content": f"你是一位数据处理专家，根据给出的信息第一步提取字段内容```{cankaoStr}```,最后给出只包含这些字段的json格式数据。"},
            {"role": "user", "content": f'内容是```{inputStr}```\n'}
        ]
    # 使用OpenAI库创建chat completion请求
    response = client.chat.completions.create(
        model=model,
        messages=message1,
        stream=False  # 这里我们不使用流式传输
    )
    # 返回响应的内容
    return response.choices[0].message.content


def save_to_excel(chat_response, excel_path):
    try:
        match = re.search(r"```json\s*(.*?)\s*```", chat_response, re.DOTALL)
        if not match:
            raise ValueError("未找到有效的 JSON 数据")
        
        json_str = match.group(1).strip()
        # 将JSON字符串加载为Python字典
        data_dict = json.loads(json_str)
        # 将字典转换为DataFrame
        df_new = pd.DataFrame([data_dict])
        
        if os.path.exists(excel_path):
            # 如果文件存在，读取现有的Excel文件
            df_existing = pd.read_excel(excel_path)
            # 合并新旧数据，保持列的一致性
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_final = df_combined.fillna('')
        else:
            df_final = df_new
        
        # 确保所有的列都在最终的DataFrame中
        all_columns = sorted(set(df_final.columns))
        df_final = df_final.reindex(columns=all_columns)
        
        # 写入Excel文件
        df_final.to_excel(excel_path, index=False, sheet_name='Sheet1', engine='openpyxl')
        print(f"数据已成功写入或追加到 {excel_path}")
    except Exception as e:
        print(f"发生错误：{e}")
        
#图像识别函数
def recognize_image(image_path):


    def image_to_base64(image_path):
        """将本地图片转换为base64字符串"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Directly use a local image file path
    image_base64 = image_to_base64(image_path)

    # Build the prompt, for demonstration, using a placeholder text.
    prompt = "识别图片中所有信息"

    # Build the full prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(image_path)

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output
    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=1024,
        num_return_sequences=1,
        do_sample=True,
    )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    return text_output

def process_image(image_file, folder_path):
    image_path = os.path.join(folder_path, image_file)
    text_output = recognize_image(image_path)
    # 调用generate_chat_completion函数并获取回复
    ai_response = generate_chat_completion(text_output)
    
    return [image_file, text_output, ai_response]




def append_to_excel(file_path, data):
    """
    将数据追加写入到指定的Excel文件中。
    
    :param file_path: Excel文件路径。
    :param data: 要写入的数据列表，每个元素代表一行数据。
    """
    with lock:  # 使用锁来保证线程安全
        df_new = pd.DataFrame([data], columns=['Image_Name', 'Text_Output', 'AI_Response'])
        
        if os.path.exists(file_path):  # 如果文件存在，以追加模式打开
            with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # 计算新数据应该插入的起始行号
                start_row = len(pd.read_excel(file_path)) + 1
                df_new.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=start_row)
        else:
            # 如果文件不存在，则创建新文件并写入表头和数据
            df_new.to_excel(file_path, index=False, sheet_name='Sheet1', engine='openpyxl')
        
        print(f"数据已成功写入或追加到 {file_path}")


                
def process_and_save(image_file, folder_path, excel_path):
    result = process_image(image_file, folder_path)  # 处理图片获取结果
    append_to_excel(excel_path, result)  # 立即追加写入Excel


def main():

    # 文件夹路径
    folder_path = r'C:\Users\admin\Desktop\模型相关\门诊病史jpg+json\门诊病史截屏'
    
    # 获取文件夹下所有.jpg图片
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # Excel文件路径
    excel_path = 'results.xlsx'
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # 根据需要调整max_workers的数量
            futures = {executor.submit(process_and_save, image_file, folder_path, excel_path): image_file for image_file in image_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='处理图片', unit='张'):
                try:
                    future.result()  # 确保我们捕获任何在process_and_save中抛出的异常
                except Exception as e:
                    print(f"处理{futures[future]}时发生错误: {e}")


if __name__ == '__main__':
    main()