import {
    Field,
    FieldDescription,
    FieldGroup,
    FieldLabel,
} from "../../ui/field";
import { Input } from "../../ui/input";
import { Button } from "../../ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";
import { ShieldCheckIcon } from "@phosphor-icons/react/dist/ssr";

export type StepFiveValueProps = {
    goldenSet: string;
    qaGate: {
        structuralValidity: number;
        domainQuality: number;
    };
}

export function SftStepFive({value, onChange, setStepActive}: {value: StepFiveValueProps, onChange: (patch: Partial<StepFiveValueProps>) => void, setStepActive: (step: number) => void}) {
    return (
		<>
			<ScrollArea className="max-h-125">
				<FieldGroup>
					<div className="flex flex-col gap-4">
						{/* Bagian Evaluation / Golden Set */}
						<div className="flex flex-col gap-2">
							<FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
								Evaluation / Golden Set
							</FieldLabel>
							<Input
								value={value.goldenSet}
								onChange={(e) =>
									onChange({ goldenSet: e.target.value })
								}
							/>
						</div>

						{/* Bagian Structural Quality & Domain Quality */}
						<div className="grid grid-cols-2 gap-3">
							<div className="flex flex-col gap-2 p-4 rounded-lg border bg-white">
								<span className="text-xs font-bold text-muted-foreground uppercase">
									Structural Quality
								</span>
								<ul className="flex flex-col gap-1.5 text-xs text-muted-foreground list-disc pl-4">
									<li>Valid output format</li>
									<li>Required fields</li>
									<li>Schema compliance</li>
								</ul>
							</div>
							<div className="flex flex-col gap-2 p-4 rounded-lg border bg-white">
								<span className="text-xs font-bold text-muted-foreground uppercase">
									Domain Quality
								</span>
								<ul className="flex flex-col gap-1.5 text-xs text-muted-foreground list-disc pl-4">
									<li>Scenario understanding</li>
									<li>Evidence grounding</li>
									<li>Hypothesis quality</li>
									<li>Recommended action quality</li>
								</ul>
							</div>
						</div>

						{/* Bagian Regression */}
						<div className="flex flex-col gap-2 p-4 rounded-lg border bg-white">
							<span className="text-xs font-bold text-muted-foreground uppercase">
								Regression
							</span>
							<ul className="flex flex-col gap-1.5 text-xs text-muted-foreground list-disc pl-4">
								<li>Golden set comparison</li>
								<li>Previous model comparison</li>
							</ul>
						</div>

						{/* Bagian Quality Gate */}
						<div className="flex flex-col gap-3 border p-4 rounded-lg bg-blue-50/10 border-blue-100">
							<div className="flex items-center gap-2">
								<ShieldCheckIcon className="h-4 w-4 text-blue-700" />
								<span className="text-xs font-bold text-blue-900 uppercase">
									Quality Gate
								</span>
							</div>
							<div className="grid grid-cols-2 gap-4">
								<Field>
									<FieldLabel className="text-xs text-muted-foreground">
										Structural Validity ≥
									</FieldLabel>
									<Input
										value={value.qaGate.structuralValidity}
										onChange={(e) =>
											onChange({
												qaGate: {
													...value.qaGate,
													structuralValidity: Number(
														e.target.value,
													),
												},
											})
										}
									/>
								</Field>
								<Field>
									<FieldLabel className="text-xs text-muted-foreground">
										Domain Quality ≥
									</FieldLabel>
									<Input
										value={value.qaGate.domainQuality}
										onChange={(e) =>
											onChange({
												qaGate: {
													...value.qaGate,
													domainQuality: Number(
														e.target.value,
													),
												},
											})
										}
									/>
								</Field>
							</div>
							<div className="flex flex-col gap-1 pt-2 border-t text-xs">
								<span className="text-muted-foreground">
									Regression score:{" "}
									<span className="font-semibold text-blue-900">
										&gt;= previous champion
									</span>
								</span>
								<span className="text-muted-foreground">
									Model promotion requires evaluation to pass
									the configured quality gate.
								</span>
							</div>
						</div>
					</div>
				</FieldGroup>
			</ScrollArea>

			<div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(4)}>
					Back
				</Button>
				<Button
					type="button"
					// disabled={!isAvailabletoNextStep()}
					// className={`${isAvailabletoNextStep() ? "" : "opacity-50 cursor-not-allowed"}`}
					onClick={() => setStepActive(6)}>
					Next
				</Button>
			</div>
		</>
	);
}