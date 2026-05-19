import os
import xml.etree.ElementTree as ET

# ===== НАСТРОЙКИ (ИЗМЕНИ ПОД СЕБЯ) =====
IMAGES_DIR = "yolo/test/images"      # папка с .jpg
LABELS_DIR = "yolo/test/labels"      # папка с .txt (YOLO)
OUTPUT_ANNOTATIONS_DIR = "voc/test/Annotations"  # куда сохранить .xml
CLASS_NAME = "cigarette"             # имя твоего класса
# =====================================

# Создаём папку для аннотаций, если её нет
os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

def convert_yolo_to_mediapipe_xml(yolo_txt_path, image_path, output_xml_path):
    # Получаем размеры изображения
    img_h, img_w = (250, 250)
    
    # Читаем YOLO аннотации
    objects = []
    with open(yolo_txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts or len(parts) < 5:
                continue
            
            class_id = int(parts[0])  # у тебя всегда 0
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            
            objects.append({
                'name': CLASS_NAME,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    if not objects:
        print(f"⚠️ Нет объектов в {yolo_txt_path}")
        return False
    
    # Создаём XML в формате MediaPipe
    annotation = ET.Element('annotation')
    
    # Добавляем filename
    filename_elem = ET.SubElement(annotation, 'filename')
    filename_elem.text = os.path.basename(image_path)
    
    # Добавляем size (опционально, но рекомендуется)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_w)
    ET.SubElement(size, 'height').text = str(img_h)
    ET.SubElement(size, 'depth').text = '3'
    
    # Добавляем объекты
    for obj in objects:
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = obj['name']
        
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])
    
    # Сохраняем XML файл
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    return True

def main():
    # Получаем список всех изображений
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Найдено изображений: {len(image_files)}")
    
    converted = 0
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        txt_path = os.path.join(LABELS_DIR, base_name + '.txt')
        img_path = os.path.join(IMAGES_DIR, img_file)
        xml_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, base_name + '.xml')
        
        if not os.path.exists(txt_path):
            print(f"⚠️ Нет .txt файла для {img_file}")
            continue
        
        if convert_yolo_to_mediapipe_xml(txt_path, img_path, xml_path):
            converted += 1
            print(f"✅ {img_file} → {base_name}.xml")
    
    print(f"\n🎉 Конвертировано {converted} из {len(image_files)} файлов")

if __name__ == "__main__":
    main()
