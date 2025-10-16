import openai
import torch
from transformers import AutoTokenizer, AutoModel
from classifier.bert_classifier import BERTClassifier, BertTokenizer
from catboost import CatBoostClassifier
import scibox_api
import nlp_entry

id_to_label_maps = {
    'Основная категория': {0: 'Новые клиенты', 1: 'Продукты - Вклады', 2: 'Продукты - Карты', 3: 'Продукты - Кредиты', 4: 'Техническая поддержка', 5: 'Частные клиенты'},
    'Подкатегория': {0: 'Автокредиты - Автокредит без залога', 1: 'Банковские карточки', 2: 'Валютные - CNY', 3: 'Валютные - EUR', 4: 'Валютные - RUB', 5: 'Валютные - USD', 6: 'Вклады и депозиты', 7: 'Дебетовые карты - Infinite', 8: 'Дебетовые карты - MORE', 9: 'Дебетовые карты - Signature', 10: 'Дебетовые карты - Комплимент', 11: 'Дебетовые карты - Форсаж', 12: 'Карты рассрочки - КСТАТИ', 13: 'Карты рассрочки - ЧЕРЕПАХА', 14: 'Кредитные карты - PLAT/ON', 15: 'Кредитные карты - Отличник', 16: 'Кредитные карты - Портмоне 2.0', 17: 'Кредиты', 18: 'Онлайн кредиты - Проще в онлайн', 19: 'Онлайн-сервисы', 20: 'Первые шаги', 21: 'Потребительские - Всё только начинается', 22: 'Потребительские - Дальше - меньше', 23: 'Потребительские - Легко платить', 24: 'Потребительские - На всё про всё', 25: 'Потребительские - Старт', 26: 'Проблемы и решения', 27: 'Регистрация и онбординг', 28: 'Рублевые - Великий путь', 29: 'Рублевые - Мои условия', 30: 'Рублевые - Мои условия онлайн', 31: 'Рублевые - Подушка безопасности', 32: 'Рублевые - СуперСемь', 33: 'Экспресс-кредиты - В магазинах-партнерах', 34: 'Экспресс-кредиты - На роднае'},
    'Целевая аудитория': {0: 'все клиенты', 1: 'новые клиенты'}
}

# model_config = {
#     'Основная категория': {
#         'path': 'models/best_model_main.bin',
#         'n_classes': len(id_to_label_maps['Основная категория'])
#     },
#     'Подкатегория': {
#         'path': 'models/best_model_sub.bin',
#         'n_classes': len(id_to_label_maps['Подкатегория'])
#     },
#     'Целевая аудитория': {
#         'path': 'models/best_model_ca.bin',
#         'n_classes': len(id_to_label_maps['Целевая аудитория'])
#     }
# }
model_config = {
    'Основная категория': 'models/catboost_model_main.cbm',
    'Подкатегория': 'models/catboost_model_sub.cbm',
    'Целевая аудитория': 'models/catboost_model_ca.cbm'
}



class FAQCatBoostInference:
    """
    Класс для инференса, использующий BERT для создания эмбеддингов
    и CatBoost для классификации.
    """

    def __init__(self, model_paths, id_to_label_maps, bert_model_name='DeepPavlov/rubert-base-cased'):
        """
        Инициализатор класса.

        Args:
            model_paths (dict): Словарь с путями к сохраненным моделям CatBoost.
                                Пример: {'Основная категория': 'path/to/model1.cbm', ...}
            id_to_label_maps (dict): Словарь с картами {id: label} для каждой категории.
            bert_model_name (str): Название предобученной BERT-модели.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство для BERT: {self.device}")

        # 1. Загрузка BERT модели и токенизатора для создания эмбеддингов
        ##self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        ##self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.pre = nlp_entry.QuestionPreprocessor(remove_stopwords=True)
        self.tokenizer = scibox_api.SciBoxClient(api_key="sk-QFTyxx7PZeJjc5cBhWygoQ")

        # 2. Загрузка обученных моделей CatBoost
        self.catboost_models = {}
        for category, path in model_paths.items():
            model = CatBoostClassifier()
            model.load_model(path)
            self.catboost_models[category] = model
            print(f"Модель CatBoost для '{category}' успешно загружена.")

        # 3. Сохранение карт для декодирования предсказаний
        self.id_to_label_maps = id_to_label_maps

    def _get_embedding(self, text):
        # processed = self.pre.preprocess(text)["normalized"]
        processed = text
        resp = self.tokenizer.embeddings(
            inputs=processed
        )
        return resp.data[0].embedding
    # def _get_embedding(self, text):
    #     """
    #     Внутренний метод для получения эмбеддинга одного текста.
    #     """
    #     inputs = self.tokenizer(
    #         text,
    #         return_tensors='pt',
    #         padding=True,
    #         truncation=True,
    #         max_length=512
    #     ).to(self.device)
    #
    #     with torch.no_grad():
    #         outputs = self.bert_model(**inputs)
    #
    #     # Используем эмбеддинг [CLS] токена
    #     cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    #     return cls_embedding

    def complete(self, question, answer):
        try:
            resp = self.tokenizer.chat("Qwen2.5-72B-Instruct-AWQ",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "Ты — «Виртуальный Консультант», эксперт по продуктам банка [Название Банка]. "
                                                       "Твоя главная цель — помогать клиентам, предоставляя точную и ясную информацию.\n\n**Основные инструкции:**\n1. "
                                                       " **Используй только контекст:** Всегда основывай свой ответ **исключительно** на информации, найденной в предоставленных документах (контексте). Но учитывай всю полноту контекста."
                                                       " Добавляй от себя только в рамках контекста. Если в контексте есть информация, о которой пользователь не спрашивает, все равно упомяни ее кратко. \n2.  **Точность и ясность:** Отвечай на вопрос пользователя прямо и по существу. "
                                                       "Если в контексте есть конкретные цифры (ставки, лимиты, сроки), обязательно их используй.\n3. "
                                                       " **Дружелюбный и профессиональный тон:** Общайся вежливо, понятно и уважительно."
                                                       " Избегай сложного банковского жаргона. Если термин сложный, объясни его простыми словами.\n\n**Строгие запреты:**\n-   **Не придумывай информацию:** Категорически запрещено домысливать детали или давать ответы, если их нет в контексте.\n-   **Не давай финансовых советов:** Не рекомендуй, какой продукт «лучше» или «выгоднее». Твоя задача — информировать, а не советовать.\n-   **Не запрашивай личные данные:** "
                                                       "Никогда не проси у пользователя ФИО, номер телефона, данные карты, паспорта и т.д."
                                                       "\n\n**Если ответа нет в контексте:**\nЕсли предоставленная информация не позволяет ответить на вопрос, используй одну из следующих формулировок:\n- "
                                                       "  «К сожалению, я не нашел точной информации по вашему вопросу. "
                                                       "Пожалуйста, попробуйте переформулировать его или обратитесь в контакт-центр банка по номеру [Номер телефона].»\n-  "
                                                       " «В моей базе знаний нет данных об этом. Для получения точной информации, пожалуйста, свяжитесь со специалистом банка. Информация (контекст) на которой нужно основывать твой ответ : {}»".format(answer)
                                        },
                                        {"role": "user", "content": question},
                                    ],
                                    temperature=0.7,
                                    top_p=0.9,
                                    max_tokens=256,
                                    )
        except openai.RateLimitError:
            return None
        return resp.choices[0].message.content
    def predict(self, text):
        """
        Делает предсказание для заданного текста по всем категориям.

        Args:
            text (str): Входной текст для классификации.

        Returns:
            dict: Словарь с предсказанными текстовыми метками.
        """
        # 1. Получаем эмбеддинг для входного текста
        embedding = self._get_embedding(text)

        predictions = {}
        # 2. Делаем предсказания для каждой категории
        for category, model in self.catboost_models.items():
            # CatBoost предсказывает индекс класса
            predicted_id = model.predict(embedding)

            # 3. Декодируем индекс в текстовую метку
            predicted_label = self.id_to_label_maps[category].get(int(predicted_id), "Неизвестный класс")
            predictions[category] = predicted_label
        predictions["embedding"] = embedding
        return predictions

class FAQInference:
    """
    Класс для инференса, который загружает обученные классификаторы
    и делает предсказания по трем категориям.
    """
    def __init__(self, model_config, id_to_label_maps, device='cpu'):
        """
        Инициализатор класса.

        Args:
            model_config (dict): Словарь с конфигурацией моделей.
                                 Пример: {'Основная категория': {'path': 'path/to/model1.bin', 'n_classes': 5}, ...}
            id_to_label_maps (dict): Словарь с маппингами id в текстовые метки.
                                     Пример: {'Основная категория': {0: 'Продукты', 1: 'Клиенты'}, ...}
            device (str): Устройство для вычислений ('cpu' или 'cuda').
        """
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.max_len = 128  # Такая же длина, как при обучении
        self.id_to_label_maps = id_to_label_maps
        self.models = {}

        # Загружаем каждую модель из конфигурации
        for category, config in model_config.items():
            model = BERTClassifier(n_classes=config['n_classes'])
            # Загружаем сохраненные веса
            model.load_state_dct(torch.load(config['path'], map_location=self.device))
            model = model.to(self.device)
            model.eval()  # Переводим модель в режим оценки
            self.models[category] = model
            print(f"Модель для '{category}' успешно загружена.")

    def predict(self, text):
        """
        Делает предсказание для заданного текста по всем категориям.

        Args:
            text (str): Входной текст для классификации (Шаблонный ответ).

        Returns:
            dict: Словарь с предсказанными категориями.
                  Пример: {'Основная категория': 'Продукты - Карты', ...}
        """
        # Токенизация входного текста
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        predictions = {}

        with torch.no_grad():
            for category, model in self.models.items():
                # Получаем "сырые" выходы модели (логиты)
                outputs = model(input_ids, attention_mask)
                # Находим индекс класса с максимальной вероятностью
                _, pred_idx = torch.max(outputs, dim=1)
                pred_idx = pred_idx.item()
                # Преобразуем индекс в текстовую метку
                predicted_label = self.id_to_label_maps[category].get(pred_idx, "Неизвестный класс")
                predictions[category] = predicted_label

        return predictions