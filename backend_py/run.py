import uvicorn
from app import app

uvicorn.run(app=app, port=3000) 