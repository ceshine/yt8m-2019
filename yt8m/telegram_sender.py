"""Source: https://github.com/huggingface/knockknock"""
import datetime
import traceback
import functools
import socket
import telegram

DATE_FORMAT = "%Y-%m-%d %H:%M:%d"


def identity(func):
    return func


def telegram_sender(token: str, chat_id: int, name: str):
    """
    Telegram sender wrapper: execute func, send a Telegram message with the end status
    (sucessfully finished or crashed) at the end. Also send a Telegram message before
    executing func.

    `token`: str
        The API access TOKEN required to use the Telegram API.
        Visit https://core.telegram.org/bots#6-botfather to obtain your TOKEN.
    `chat_id`: int
        Your chat room id with your notification BOT.
        Visit https://api.telegram.org/bot<YourBOTToken>/getUpdates to get your chat_id
        (start a conversation with your bot by sending a message and get the `int` under
        message['chat']['id'])
    """
    if token == "":
        return identity

    bot = telegram.Bot(token=token)

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__
            contents = [f'{name} has started 🎬',
                        'Machine name: %s' % host_name,
                        'Main call: %s' % func_name,
                        'Starting date: %s' % start_time.strftime(DATE_FORMAT)]
            text = '\n'.join(contents)
            bot.send_message(chat_id=chat_id, text=text)

            try:
                value = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = [f"{name} is complete 🎉",
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(
                                DATE_FORMAT),
                            'End date: %s' % end_time.strftime(DATE_FORMAT),
                            'Training duration: %s' % str(elapsed_time)]
                text = '\n'.join(contents)
                bot.send_message(chat_id=chat_id, text=text)
                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["Your training has crashed ☠️",
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(
                                DATE_FORMAT),
                            'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            'Crashed training duration: %s\n\n' % str(
                                elapsed_time),
                            "Here's the error:",
                            '%s\n\n' % ex,
                            "Traceback:",
                            '%s' % traceback.format_exc()[:300]]
                text = '\n'.join(contents)
                bot.send_message(chat_id=chat_id, text=text)
                raise ex

        return wrapper_sender

    return decorator_sender
