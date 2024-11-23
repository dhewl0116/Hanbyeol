import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  Request,
  UseGuards,
} from '@nestjs/common';
import { MatchService } from './match.service';
import { JWTAuthGuard } from 'src/auth/Guards/jwt.auth-guard';
import { EditProgressDTO, LawyerDTO } from './dto/match.dto';

@UseGuards(JWTAuthGuard)
@Controller('match')
export class MatchController {
  constructor(private matchService: MatchService) {}

  @Get('check_matched_lawyer')
  async checkMatchedLawyer(@Request() req) {
    const { userId: applicantId } = req.user;
    return await this.matchService.checkMatchedLawyer(applicantId);
  }

  @Get(':id')
  async getMatch(@Param('id') opponentId: string, @Request() req) {
    const { userId, role } = req.user;
    if (role === 'applicant') {
      return await this.matchService.getMatch(userId, opponentId);
    } else if (role === 'lawyer') {
      return await this.matchService.getMatch(opponentId, userId);
    }
  }

  @Post('send_request')
  async sendRequest(@Body() lawyerDTO: LawyerDTO, @Request() req) {
    const { userId: applicantId } = req.user;
    return await this.matchService.sendRequest(applicantId, lawyerDTO);
  }

  @Patch('edit_progress/:id')
  async editProgress(
    @Param('id') id: string,
    @Body() editProgressDTO: EditProgressDTO,
    @Request() req,
  ) {
    const { userId: lawyerId } = req.user;

    return await this.matchService.editProgress(id, lawyerId, editProgressDTO);
  }

  @Delete('reject_request/:id')
  async rejectRequest(@Param('id') id: string, @Request() req) {
    const { userId: lawyerId } = req.user;
    return await this.matchService.rejectRequest(id, lawyerId);
  }
}
