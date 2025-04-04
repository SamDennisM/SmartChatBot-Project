import ollama

def chat_with_ollama(model="mistral"):
    print(f"{model.capitalize()} Chatbot (Type 'exit' to quit)")
    chat_history = []  # Stores conversation context
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a valid message.")
                continue

            chat_history.append({"role": "user", "content": user_input})
            response = ollama.chat(model=model, messages=chat_history)
            
            bot_reply = response.get('message', {}).get('content', "[No response received]")
            chat_history.append({"role": "assistant", "content": bot_reply})
            
            print("Bot:", bot_reply)
        
        except ollama.exceptions.OllamaError as e:
            print("Ollama Error:", e)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("Unexpected Error:", e)
            break

if __name__ == "__main__":
    chat_with_ollama()