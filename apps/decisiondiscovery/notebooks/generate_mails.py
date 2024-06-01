#%%
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

ongoing_mail = "dscontroldev@gmail.com"

#%%
def send_email(name, surname, registration_date, id_card, tax_id, bank_account, email, address, doc_145, company):
    # Configuración del servidor SMTP
    smtp_server = 'smtp.freesmtpservers.com'
    smtp_port = 25
    smtp_user = 'rpaustester@test.com'
    smtp_password = 'None'
    
    # Crear el mensaje
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = ongoing_mail  # o el email del administrativo si se envía a él
    msg['Subject'] = f'Registration as employee - {name} {surname}'
    
    # Plantilla del cuerpo del mensaje
    body = f"""\
    Dear Administration Dept.,

    I am pleased to submit my personal details for the registration process as a new employee of {company}. Below, you will find my detailed information:

    - **Full Name**: {name} {surname}
    - **ID Card Number**: {id_card}
    - **Tax ID**: {tax_id}
    - **Bank Account**: {bank_account}
    - **Address**: {address}

    Best regards,

    {name} {surname}
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Iniciar sesión en el servidor SMTP smtp.freesmtpservers.com en el puerto 25 y sin autenticación
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # server.starttls()
        # server.login(smtp_user, smtp_password)
        server.login(smtp_user, None)
        server.send_message(msg)
        
    print(f'Email enviado a {email}')
    
#%%

# Suponiendo que la función send_email ya está definida como se mostró anteriormente.

# Cambia esto por la ruta real a tu archivo CSV
file_path = '../../../resources/workers_registration.csv'

# Leer el archivo CSV
data = pd.read_csv(file_path, sep=';')

# Iterar a través de cada fila del DataFrame
for index, row in data.iterrows():
    send_email(
        name=row['NAME'],
        surname=row['SURNAME'],
        registration_date=row['REGISTRATION DATE'],
        id_card=row['ID CARD'],
        tax_id=row['TAX ID'],
        bank_account=row['BANK ACCOUNT'],
        email=row['EMAIL'],
        address=row['ADDRESS'],
        doc_145=row['DOC. 145'],
        company=row['COMPANY']
    )

# %%
