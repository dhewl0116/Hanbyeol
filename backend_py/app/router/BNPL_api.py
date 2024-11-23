from fastapi import APIRouter, HTTPException
from app.router.DTO import BNPL_DTO, BNPL_return_DTO
from app.service import BNPL_administration
from app.model import db_handler

bnpl_router = APIRouter(prefix='/bnpl')
bnpl_administration = BNPL_administration("id", "key")


@bnpl_router.post('/call')
async def bnpl_call(param:BNPL_DTO):
    user_id, account_num, money = param.model_dump().values()
    
    current_amount = await db_handler.get_bnpl_amount(user_id)
    
    if current_amount is None:
        raise HTTPException(status_code=404, detail="유저의 대출 기록을 찾을 수 없습니다.")
    
    if current_amount + money > 300000:
        raise HTTPException(status_code=400, detail="대출 한도를 초과합니다.")
    
    bnpl_administration.virtual_autotransfer(account_num, money)

    db_handler.update_bnpl_amount(user_id, current_amount + money)

    return {"status": "success"}

@bnpl_router.post('/remove')
async def remove_bnpl_amount(param:BNPL_return_DTO):
    user_id, money = param.model_dump().values()

    current_amount = await db_handler.get_bnpl_amount(user_id)
    
    if current_amount is None:
        raise HTTPException(status_code=404, detail="유저의 대출 기록을 찾을 수 없습니다.")
    
    if current_amount > 300000:
        raise HTTPException(status_code=400, detail="금액이 0보다 작아집니다.")
    
    db_handler.update_bnpl_amount(user_id, current_amount - money)

    return {"status": "success"}

@bnpl_router.post("/user")
async def get_money_from_user(param):
    user_id = param.user_id

    return await db_handler.get_bnpl_amount(user_id)

    

     

