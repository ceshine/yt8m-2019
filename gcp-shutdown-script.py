#!/home/ceshine/miniconda3/envs/pytorch/bin/python
import socket
import telegram

BOT_TOKEN = ""
CHAT_ID = ""

bot = telegram.Bot(token=BOT_TOKEN)
host_name = socket.gethostname()
content = 'Machine name: %s is shutting down!' % host_name
bot.send_message(chat_id=CHAT_ID, text=content)
