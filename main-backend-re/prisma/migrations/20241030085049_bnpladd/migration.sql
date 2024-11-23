/*
  Warnings:

  - You are about to drop the column `participant_id` on the `Match` table. All the data in the column will be lost.
  - The `role` column on the `User` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - Added the required column `applicant_id` to the `Match` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "Match" DROP CONSTRAINT "Match_participant_id_fkey";

-- AlterTable
ALTER TABLE "Match" DROP COLUMN "participant_id",
ADD COLUMN     "applicant_id" TEXT NOT NULL;

-- AlterTable
ALTER TABLE "User" DROP COLUMN "role",
ADD COLUMN     "role" TEXT NOT NULL DEFAULT 'unknown';

-- DropEnum
DROP TYPE "Role";

-- CreateTable
CREATE TABLE "Bnpl" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "amount" INTEGER NOT NULL,

    CONSTRAINT "Bnpl_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Match" ADD CONSTRAINT "Match_applicant_id_fkey" FOREIGN KEY ("applicant_id") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Bnpl" ADD CONSTRAINT "Bnpl_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
