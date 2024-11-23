/*
  Warnings:

  - A unique constraint covering the columns `[applicant_id,lawyer_id]` on the table `Match` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "Match_applicant_id_lawyer_id_key" ON "Match"("applicant_id", "lawyer_id");
