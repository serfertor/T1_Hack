import torch

from bert_classifier import *

id_to_label_maps = {
    'Основная категория': {
        0: 'Новые клиенты',
        1: 'Продукты - Вклады',
        2: 'Продукты - Карты',
        3: 'Продукты - Кредиты',
        4: 'Техническая поддержка',
        5: 'Частные клиенты'
    },
    'Подкатегория': {
        0: 'Автокредиты - Автокредит без залога',
        1: 'Банковские карточки',
        2: 'Валютные - CNY',
        3: 'Валютные - EUR',
        4: 'Валютные - RUB',
        5: 'Валютные - USD',
        6: 'Вклады и депозиты',
        7: 'Дебетовые карты - MORE',
        8: 'Дебетовые карты - Signature',
        9: 'Дебетовые карты - Infinite',
        10: 'Дебетовые карты - Комплимент',
        11: 'Дебетовые карты - Форсаж',
        12: 'Карты рассрочки - КСТАТИ',
        13: 'Карты рассрочки - ЧЕРЕПАХА',
        14: 'Кредиты',
        15: 'Кредитные карты - PLAT/ON',
        16: 'Кредитные карты - Отличник',
        17: 'Кредитные карты - Портмоне 2.0',
        18: 'Онлайн кредиты - Проще в онлайн',
        19: 'Онлайн-сервисы',
        20: 'Первые шаги',
        21: 'Потребительские - Всё только начинается',
        22: 'Потребительские - Дальше - меньше',
        23: 'Потребительские - Легко платить',
        24: 'Потребительские - На всё про всё',
        25: 'Потребительские - Старт',
        26: 'Проблемы и решения',
        27: 'Регистрация и онбординг',
        28: 'Рублевые - Великий путь',
        29: 'Рублевые - Мои условия',
        30: 'Рублевые - Мои условия онлайн',
        31: 'Рублевые - Подушка безопасности',
        32: 'Рублевые - СуперСемь',
        33: 'Экспресс-кредиты - В магазинах-партнерах',
        34: 'Экспресс-кредиты - На роднае'
    },
    'Целевая аудитория': {
        0: 'все клиенты',
        1: 'новые клиенты'
    }
}

model_config = {
    'Основная категория': {
        'path': 'best_model_main.bin',
        'n_classes': len(id_to_label_maps['Основная категория'])
    },
    'Подкатегория': {
        'path': 'best_model_sub.bin',
        'n_classes': len(id_to_label_maps['Подкатегория'])
    },
    'Целевая аудитория': {
        'path': 'best_model_ca.bin',
        'n_classes': len(id_to_label_maps['Целевая аудитория'])
    }
}

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
            model.load_state_dict(torch.load(config['path'], map_location=self.device))
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