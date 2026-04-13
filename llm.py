from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings('ignore')

class ModelInterface:
    def __init__(self, model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Загрузка TinyLlama-1.1B на устройство: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.token_processor = AutoTokenizer.from_pretrained(model_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.language_model = self.language_model.to(self.device)
        self.language_model.eval()
        
        if self.token_processor.pad_token is None:
            self.token_processor.pad_token = self.token_processor.eos_token
        
        print("Модель успешно загружена!")
    
    def get_response(self, user_input: str, max_length: int = 180) -> str:
        try:
            # Улучшенный промпт для развёрнутых ответов
            formatted_prompt = f"""<|system|>
Ты полезный ассистент. Отвечай подробно и развёрнуто. Объясняй детали. Дай полный ответ на вопрос.</s>
<|user|>
{user_input}</s>
<|assistant|>
"""
            
            input_tensors = self.token_processor(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
            
            with torch.no_grad():
                generated_outputs = self.language_model.generate(
                    **input_tensors,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.15,
                    pad_token_id=self.token_processor.pad_token_id,
                    eos_token_id=self.token_processor.eos_token_id,
                    use_cache=True
                )
            
            response_tokens = generated_outputs[0][len(input_tensors['input_ids'][0]):]
            final_response = self.token_processor.decode(
                response_tokens,
                skip_special_tokens=True
            ).strip()
            
            if not final_response:
                return "Извините, не могу ответить на этот вопрос."
            
            # Убираем возможные повторения
            if len(final_response) > 500:
                final_response = final_response[:500] + "..."
            
            return final_response
            
        except Exception as e:
            print(f"Ошибка при генерации: {e}")
            return f"Произошла ошибка: {str(e)}"