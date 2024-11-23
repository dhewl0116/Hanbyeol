import {
  Body,
  Controller,
  Get,
  Post,
  Request,
  UseGuards,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { LocalAuthGuard } from './Guards/local.auth-guard';
import { JWTAuthGuard } from './Guards/jwt.auth-guard';
import { RegisterDTO } from './dto/auth.dto';

@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @UseGuards(LocalAuthGuard)
  @Post('login')
  async login(@Request() req) {
    return this.authService.login(req.user);
  }

  @Post('register')
  async register(@Body() registerDTO: RegisterDTO) {
    return this.authService.register(registerDTO);
  }

  @UseGuards(JWTAuthGuard)
  @Get('profile')
  async GetProfile(@Request() req) {
    return req.user;
  }
}
