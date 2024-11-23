// chat.service.ts
import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';
import { SendChatDTO } from './dto/chat.dto';

@Injectable()
export class ChatService {
  constructor(private readonly prisma: PrismaService) {}

  async get_chat(match_id: string) {
    return await this.prisma.message.findMany({
      where: { match_id },
      orderBy: { created_at: 'asc' },
    });
  }

  async save_chat(user_id: string, sendChatDTO: SendChatDTO) {
    return await this.prisma.message.create({
      data: {
        match_id: sendChatDTO.match_id,
        user_id,
        content: sendChatDTO.content,
      },
    });
  }
}
