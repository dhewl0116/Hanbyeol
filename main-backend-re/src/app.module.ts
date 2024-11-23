import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { PrismaService } from './prisma/prisma.service';
import { UserModule } from './user/user.module';
import { MatchModule } from './match/match.module';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [AuthModule, UserModule, MatchModule, ChatModule],
  controllers: [AppController],
  providers: [AppService, PrismaService],
})
export class AppModule {}
