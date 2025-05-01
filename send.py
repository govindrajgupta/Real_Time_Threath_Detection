from twilio.rest import Client
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)
def sendSms():
    message = client.messages.create(
    from_='number',
    body='Alert: Human detected in the camera feed.',
    to='receiver number'
)
    print(message.sid)