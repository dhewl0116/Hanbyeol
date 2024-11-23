import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';

@Injectable()
export class UserService {
  constructor(private readonly prisma: PrismaService) {}

  async getUserById(id: string) {
    return await this.prisma.user.findUniqueOrThrow({
      where: { id },
      select: {
        id: true,
        username: true,
        role: true,
        email: true,
        description: true,
        created_at: true,
      },
    });
  }

  async getApplicants(lawyerId: string) {
    // 변호사 아이디로 신청 대기 중인 사람 찾기
    return await this.prisma.match.findMany({
      where: {
        lawyer_id: lawyerId,
      },
      include: {
        applicant: true,
      },
    });
  }
}
