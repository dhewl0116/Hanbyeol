from requests.auth import HTTPBasicAuth
import requests
import time

class BNPL_administration:
    def __init__(self, id="YOUR_CLIENT_ID", secret="YOUR_CLIENT_SECRET"):
        self.client_id = id
        self.client_secret = secret
        
    def get_access_token(self):
        # OAuth 2.0을 이용한 액세스 토큰 발급 요청
        token_url = "https://api.kftc.or.kr/oauth/2.0/token"  # 금융결제원 토큰 발급 URL (가정)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post(token_url, headers=headers, data=payload)
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            print("Access token 발급 성공:", access_token)
            return access_token
        else:
            print("Access token 발급 실패:", response.text)
            return None

    def transfer_funds(self, account_number, money, access_token):
        # 송금 요청
        transfer_url = "https://api.kftc.or.kr/v1/transfer"  # 금융결제원 송금 URL (가정)
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "account_number": account_number,
            "amount": money,
            "currency": "KRW"  # 통화 코드 (가정)
        }

        response = requests.post(transfer_url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"송금 성공: {account_number} 계좌로 {money}원이 송금되었습니다.")
            return True
        else:
            print("송금 실패:", response.text)
            return False

    def virtual_autotransfer(self, account_number, money):
        # Access Token 발급 요청
        access_token = self.get_access_token()
        if not access_token:
            print("액세스 토큰 발급 실패로 송금이 중단되었습니다.")
            return False
        
        # 1초 대기 (송금 전 대기)
        time.sleep(1)
        
        # 송금 요청
        success = self.transfer_funds(account_number, money, access_token)
        return success
