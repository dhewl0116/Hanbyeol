import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';
import { EditProgressDTO, LawyerDTO } from './dto/match.dto';

@Injectable()
export class MatchService {
  constructor(private prisma: PrismaService) {}

  async checkMatchedLawyer(applicantId: string) {
    const res = await this.prisma.match.findFirst({
      where: {
        applicant_id: applicantId,
      },
      include: {
        lawyer: true,
      },
    });
    console.log(res);
    return res;
  }

  async getMatch(applicantId: string, lawyerId: string) {
    // for applicant
    return await this.prisma.match.findUnique({
      where: {
        applicant_id_lawyer_id: {
          applicant_id: applicantId,
          lawyer_id: lawyerId,
        },
      },
    });
  }

  async sendRequest(applicantId: string, lawyerDTO: LawyerDTO) {
    // for applicant
    return await this.prisma.match.create({
      data: {
        applicant_id: applicantId,
        lawyer_id: lawyerDTO.lawyerId,
        // status: not accepted 는 기본값이기 때문에 안넣어도됨
      },
    });
  }

  // 요청 승인도 이 엔드포인트 따라가면 됨.
  async editProgress(
    applicantId: string,
    lawyerId: string,
    editProgressDTO: EditProgressDTO,
  ) {
    return await this.prisma.match.update({
      where: {
        applicant_id_lawyer_id: {
          applicant_id: applicantId,
          lawyer_id: lawyerId,
        },
      },
      data: {
        status: editProgressDTO.status,
        // 필요서류 준비, 보정권고 대응, 채권자 집회, 변제계획 인가
        // document submission
        // correction response
        // creditor meeting
        // repayment plan approval
      },
    });
  }

  async rejectRequest(applicantId: string, lawyerId: string) {
    return await this.prisma.match.delete({
      where: {
        applicant_id_lawyer_id: {
          applicant_id: applicantId,
          lawyer_id: lawyerId,
        },
      },
    });
  }
}
