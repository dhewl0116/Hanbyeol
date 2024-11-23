import { IsNotEmpty, IsString } from 'class-validator';

export class LawyerDTO {
  @IsNotEmpty()
  @IsString()
  lawyerId: string;
}

export class EditProgressDTO {
  @IsNotEmpty()
  @IsString()
  status: string;
}
