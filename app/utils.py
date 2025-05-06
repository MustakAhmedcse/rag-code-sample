import re

def is_bangla(text: str) -> bool:
    bangla_pattern = re.compile(r'[\u0980-\u09FF]')
    return bool(bangla_pattern.search(text))

def is_relevant_to_retailer_app(question: str) -> bool:
    retailer_keywords = [
        "retailer", "banglalink", "app", "profile", "device", "registration",
        "feedback", "voice of retailer", "commission", "lifting", "dashboard",
        "login", "logout", "edit", "submit", "form", "operator", "category"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in retailer_keywords)
