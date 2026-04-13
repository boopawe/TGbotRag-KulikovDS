import chromadb
import uuid

class KnowledgeBase:
    def __init__(self, storage_path="./vector_store", collection_title="knowledge_base"):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection_name = collection_title
        self._initialize_collection()
    
    def _initialize_collection(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)
            self._load_demo_data()
    
    def _load_demo_data(self):
        demo_facts = [
            'Масса сверхмассивной чёрной дыры в центре Млечного Пути (Стрелец A*) составляет около 4 миллионов масс Солнца. Эта чёрная дыра находится на расстоянии 26 тысяч световых лет от Земли.',
            'Самая быстро вращающаяся нейтронная звезда (пульсар PSR J1748-2446ad) делает 716 оборотов в секунду. Её экваториальная скорость достигает 24% от скорости света.',
            'Расстояние до ближайшей к Солнцу звезды (Проксима Центавра) составляет 4,24 световых года. Это примерно 40 триллионов километров.',
            'Возраст Вселенной, согласно последним данным спутника Planck, составляет 13,8 миллиардов лет. Вселенная начала своё существование после Большого взрыва.',
            'Гамма-всплески - самые мощные взрывы во Вселенной, за секунду они выделяют столько же энергии, сколько Солнце выделяет за всю свою жизнь (около 10 миллиардов лет).'
        ]
        
        for fact in demo_facts:
            self.store_information(fact)
    
    def store_information(self, text_content: str) -> str:
        record_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text_content],
            ids=[record_id]
        )
        return record_id
    
    def find_similar(self, query_text: str, results_count: int = 2) -> str:
        search_results = self.collection.query(
            query_texts=[query_text], 
            n_results=results_count
        )
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return ""
        
        # Объединяем несколько найденных фактов
        return "\n".join(search_results['documents'][0])
    
    def get_all_records(self) -> str:
        all_data = self.collection.get()
        
        if not all_data['documents']:
            return ""
        
        formatted_records = []
        for idx, record in enumerate(all_data['documents'], 1):
            formatted_records.append(f"{idx}. {record}")
        
        return "\n".join(formatted_records)
    
    def get_records_count(self) -> int:
        return len(self.collection.get()['documents'])
    
    def clear_all_records(self) -> None:
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)