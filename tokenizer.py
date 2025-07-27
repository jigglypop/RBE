#!/usr/bin/env python3
import sys
from transformers import PreTrainedTokenizerFast

def main():
    if len(sys.argv) < 3:
        print("Usage: python tokenizer.py [encode|decode] <text>")
        sys.exit(1)
    
    command = sys.argv[1]
    text = " ".join(sys.argv[2:])
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    
    if command == "encode":
        tokens = tokenizer.encode(text)
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)}")
    elif command == "decode":
        # Assume text is a comma-separated list of token IDs
        try:
            tokens = [int(t.strip()) for t in text.split(",")]
            decoded = tokenizer.decode(tokens)
            print(f"Decoded: {decoded}")
        except:
            print("Error: Invalid token format. Use comma-separated integers.")
    else:
        print("Unknown command. Use 'encode' or 'decode'")

if __name__ == "__main__":
    main() 