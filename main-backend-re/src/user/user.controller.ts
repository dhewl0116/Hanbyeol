import { Controller, Request, Get, UseGuards, Param } from '@nestjs/common';
import { UserService } from './user.service';
import { JWTAuthGuard } from 'src/auth/Guards/jwt.auth-guard';

@UseGuards(JWTAuthGuard)
@Controller('user')
export class UserController {
  constructor(private userService: UserService) {}

  @Get('get_applicants') // only for lawyer
  async getUnAcceptedParticipants(@Request() req) {
    const { userId: lawyerId } = req.user;

    const res = await this.userService.getApplicants(lawyerId);
    console.log(res);
    return res;
  }

  @Get(':id')
  async getUserById(@Param('id') id: string) {
    return this.userService.getUserById(id);
  }
}
