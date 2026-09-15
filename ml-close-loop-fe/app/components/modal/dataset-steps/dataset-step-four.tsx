import { CircleCheck, CirclePlus, FileWarning, Shield } from "lucide-react";
import { useState } from "react";
import { columnsRecordSample } from "~/components/table/columns/sample-record-column";
import { DataTable } from "~/components/table/data-table";
import { Button } from "~/components/ui/button";
import { FieldGroup, FieldLabel } from "~/components/ui/field";
import { Separator } from "~/components/ui/separator";
import { checks } from "~/utils";

export type StepFourValueProps = {
    detectedSample: number;
    status: "pass" | "fail";
}

export function DatasetStepFour({value, setStepActive}: {
    value: StepFourValueProps;
    setStepActive: (step: number) => void;
}) {

    const qualityReport = [
        { label: "Valid Records", value: value.detectedSample ?? 0, className: "border border-green-500 text-green-500 rounded-md bg-green-50" },
        { label: "Warnings", value: 0, className: "border border-yellow-500 text-yellow-500 rounded-md bg-yellow-50" },
        { label: "Errors", value: 0, className: "border border-slate-500 text-slate-500 rounded-md bg-slate-50" },
    ];

    const gateStatus = value.status ?? "fail";

    return (
        <div>
            <FieldGroup>
                <div className="flex flex-col gap-4 border p-4 rounded-md">
                    <div className="flex w-full justify-between items-center">
                        <div className="flex gap-1">
                            <Shield className="w-4 h-4" />
                            <p>Pre-Training Validation Gate</p>
                        </div>
                        <div className={`flex items-center gap-1 border px-1 py-0.5 rounded-sm bg-red-50 ${gateStatus === "pass" ? "border-green-500 bg-green-50 text-green-500" : "border-red-500 bg-red-50 text-red-500"}`}>
                            <CirclePlus className="w-3 h-3 rotate-45" />
                            <span className="text-sm font-semibold">{gateStatus === "pass" ? "Pass" : "Fail"}</span>
                        </div>
                    </div>

                    <Separator />

                    <div className="grid md:grid-cols-3 sm:grid-cols-2 gap-4 mt-4">
                        {checks.map((check) => (
                            <div key={check.id} className={`${check.status === "success" ? "text-green-500" : "text-red-500"} flex items-center gap-2`}>
                                {check.status === "success" ? (
                                    <CircleCheck className="w-4 h-4" />
                                ) : (
                                    <CirclePlus className="w-4 h-4 rotate-45" />
                                )}
                                <FieldLabel>{check.title}</FieldLabel>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="grid md:grid-cols-3 gap-4">
                    {qualityReport.map((report) => (
                        <div key={report.label} className={`flex flex-col w-full border-b p-2 ${report.className}`}>
                            <span className="text-sm  font-bold">{report.label}</span>
                            <span className="text-lg  font-bold">{report.value}</span>
                        </div>
                    ))}
                </div>
                <div className="flex gap-1 items-center">
                    <FileWarning className="w-4 h-4 text-orange-500" />
                    <span className="text-muted-foreground">View Validation Diagnostics ({qualityReport[1].value} notices, {qualityReport[2].value} errors)</span>
                </div>
                <DataTable columns={columnsRecordSample} data={[]} pagination={false} />
            </FieldGroup>
            <div className="flex justify-between items-center gap-2 mt-4">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(3)}>
					Back
				</Button>
				<Button
					type="button"
					onClick={() => setStepActive(5)}>
					Next
				</Button>
			</div>
        </div>
    )
}