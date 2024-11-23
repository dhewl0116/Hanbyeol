import { IsNotEmpty, IsString } from 'class-validator';

export class SendRequestDTO {
  @IsString()
  @IsNotEmpty()
  lawyerId: string;
}
