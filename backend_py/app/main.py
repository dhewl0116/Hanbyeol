from fastapi import FastAPI
from app.router import *  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(AI_router) 
app.include_router(bnpl_router) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
