/*
  Warnings:

  - You are about to drop the column `bio` on the `User` table. All the data in the column will be lost.
  - You are about to drop the `Story` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "Story" DROP CONSTRAINT "Story_user_id_fkey";

-- AlterTable
ALTER TABLE "User" DROP COLUMN "bio",
ADD COLUMN     "description" TEXT NOT NULL DEFAULT '';

-- DropTable
DROP TABLE "Story";

-- CreateTable
CREATE TABLE "Vector" (
    "id" TEXT NOT NULL,
    "vector" DOUBLE PRECISION[],
    "user_id" TEXT NOT NULL,

    CONSTRAINT "Vector_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Vector" ADD CONSTRAINT "Vector_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
