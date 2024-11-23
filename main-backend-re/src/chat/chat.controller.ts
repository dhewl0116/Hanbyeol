// chat.controller.ts
import {
  Controller,
  Get,
  Post,
  Body,
  Request,
  UseGuards,
  Param,
} from '@nestjs/common';
import { ChatService } from './chat.service';
import { JWTAuthGuard } from 'src/auth/Guards/jwt.auth-guard';
import { SendChatDTO } from './dto/chat.dto';

@UseGuards(JWTAuthGuard)
@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Get('get_messages/:id')
  async get_chat(@Param('id') id: string) {
    return await this.chatService.get_chat(id);
  }

  @Post('send_message')
  async save_chat(@Body() sendChatDTO: SendChatDTO, @Request() req) {
    const { userId: user_id } = req.user;
    return await this.chatService.save_chat(user_id, sendChatDTO);
  }
}
