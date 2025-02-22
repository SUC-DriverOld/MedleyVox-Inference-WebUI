import json
import locale
import os

class I18nAuto:
    def __init__(self, language):
        if not language or language == "auto":
            language = locale.getdefaultlocale()[0]
        if not os.path.exists(f"utils/locale/{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = self.load_language_list()

    def __call__(self, key):
        return self.language_map.get(key, key)

    def load_language_list(self):
        with open(f"utils/locale/{self.language}.json", "r", encoding="utf-8") as f:
            language_list = json.load(f)
        return language_list


def extract_i18n_strings(file_content):
    import re
    pattern = re.compile(r'i18n\("([^"]+)"\)')
    return pattern.findall(file_content)

def process_py_file(file_path):
    i18n_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        i18n_strings = extract_i18n_strings(content)
        for string in i18n_strings:
            i18n_dict[string] = ""
    return i18n_dict


if __name__ == "__main__":
    input_paths = ["E:/AI/MedleyVox/webui.py"]
    targets = ["utils/locale/en_US.json","utils/locale/zh_CN.json"]
    template = "utils/locale/template.json"

    i18n_dict = {}
    for path in input_paths:
        if os.path.isfile(path):
            file_dict = process_py_file(path)
            i18n_dict.update(file_dict)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        file_dict = process_py_file(file_path)
                        i18n_dict.update(file_dict)
    with open(template, 'w', encoding='utf-8') as f:
        json.dump(i18n_dict, f, ensure_ascii=False, indent=4)
    for target in targets:
        template_key = i18n_dict
        if "en_US" in target:
            en_us_data = {k: k for k in template_key.keys()}
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(en_us_data, f, ensure_ascii=False, indent=4)
            continue
        try:
            with open(target, 'r', encoding='utf-8') as f:
                target_key = json.load(f)
        except:
            target_key = {}
        for key in template_key:
            if key in target_key:
                template_key[key] = target_key[key]
            else:
                print("Missing: " + key)
        old, new = {}, {}
        for key in template_key:
            if template_key[key] == "":
                new[key] = template_key[key]
            else:
                old[key] = template_key[key]
        template_key = {**old, **new}
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(template_key, f, ensure_ascii=False, indent=4)