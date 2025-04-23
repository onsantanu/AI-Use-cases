from ocr.extractor import extract_text_from_image
from parser.instruction_parser import parse_instruction
from executor.action_handler import execute_action

if __name__ == "__main__":
    img_path = "samples/instruction.png"  # Path to your image file
    text = extract_text_from_image(img_path)
    print("📝 OCR Text:\n", text)

    commands = parse_instruction(text)
    print("✅ Parsed Commands:", commands)

    for cmd in commands:
        execute_action(cmd)