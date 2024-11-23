import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcryptjs';
import { PrismaService } from '../prisma/prisma.service';
import { RegisterDTO } from './dto/auth.dto';

@Injectable()
export class AuthService {
  constructor(
    private jwtService: JwtService,
    private prisma: PrismaService,
  ) {}

  async validateUser(email: string, password: string): Promise<any> {
    const user = await this.prisma.user.findUnique({ where: { email } });
    if (user && (await bcrypt.compare(password, user.password))) {
      return user;
    }
    throw new UnauthorizedException('Invalid credentials');
  }

  async login(user: any) {
    const payload = { username: user.username, sub: user.id };
    return {
      access_token: this.jwtService.sign(payload),
    };
  }

  async register(registerDTO: RegisterDTO) {
    const existingUser = await this.prisma.user.findUnique({
      where: { email: registerDTO.email },
    });

    if (existingUser) {
      throw new Error('이미 등록된 이메일입니다.');
    }
    const hashedPassword = await bcrypt.hash(registerDTO.password, 10);
    const user = await this.prisma.user.create({
      data: {
        ...registerDTO,
        password: hashedPassword,
      },
    });
    console.log(user);
    return this.login(user);
  }
}
