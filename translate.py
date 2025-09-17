# translate.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"  # pretrained English->Hindi


def load_model(model_name=MODEL_NAME, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device


def translate_texts(texts, tokenizer, model, device, max_length=128, num_beams=4):
    # Tokenize
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
        )
    # Decode
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="English -> Hindi translator (PyTorch + transformers)"
    )
    parser.add_argument(
        "--text", "-t", type=str, help="Single sentence to translate (wrap in quotes)."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to a file with one sentence per line to translate.",
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=8, help="Batch size for file input."
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    if args.text:
        outs = translate_texts([args.text], tokenizer, model, device)
        print("EN -> HI")
        print("EN:", args.text)
        print("HI:", outs[0])
    elif args.file:
        # read file and translate in batches
        lines = [l.strip() for l in open(args.file, "r", encoding="utf-8") if l.strip()]
        for i in range(0, len(lines), args.batch):
            batch = lines[i : i + args.batch]
            outs = translate_texts(batch, tokenizer, model, device)
            for en, hi in zip(batch, outs):
                print("EN:", en)
                print("HI:", hi)
                print("-" * 40)
    else:
        # interactive mode
        print("Interactive mode. Type English sentence (empty line to exit).")
        while True:
            s = input("EN> ").strip()
            if not s:
                break
            out = translate_texts([s], tokenizer, model, device)[0]
            print("HI>", out)


if __name__ == "__main__":
    main()
