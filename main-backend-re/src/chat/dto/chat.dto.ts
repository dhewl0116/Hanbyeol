import { IsNotEmpty, IsString } from 'class-validator';

export class SendChatDTO {
  @IsNotEmpty()
  @IsString()
  match_id: string;

  @IsNotEmpty()
  @IsString()
  content: string;
}
