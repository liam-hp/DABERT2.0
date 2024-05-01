def cprint(message:str, m_type:str):
  # emojis: https://emojicopy.com/

  typeToEmoji = {
    "info": "📰",
    "test": "🧪",
    "search": "🔎",
    "warn": "⚠️",
    "error": "👺",
    "success": "✅",
    "important": "❗",
    "try": "🙏",
    "wait": "⏳",

    "save": "💾",
    "setting": "⚙️",

    None: "🤷",
  }
  print(f"{typeToEmoji[m_type]}: {message}")