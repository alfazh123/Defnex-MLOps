import { Field, FieldGroup, FieldLabel } from "../../ui/field";
import { Button } from "../../ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "../../ui/select";
import { ScrollArea } from "~/components/ui/scroll-area";
import { datasetForSft } from "~/utils";
import { CircleWavyCheckIcon, LockIcon } from "@phosphor-icons/react/dist/ssr";

export type StepOneValueProps = {
    dataset: string;
    datasetVersion: string;
}

export function SftStepOne({value, onChange, setStepActive, toggleModal }: {value: StepOneValueProps, onChange: (patch: Partial<StepOneValueProps>) => void, setStepActive: (step: number) => void, toggleModal: () => void}) {
    return (
		<>
			<ScrollArea className="max-h-125">
				<FieldGroup>
					<div className="flex flex-col gap-4">
						{/* Bagian Select Dataset & Immutable Version */}
						<div className="grid grid-cols-2 gap-4">
							<Field>
								<FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
									Dataset
								</FieldLabel>
								<Select
									value={value.dataset}
									onValueChange={(val) =>
										onChange({
											dataset: val ?? "",
											datasetVersion:
												datasetForSft.find(
													(data) =>
														data.value === val,
												)?.versions[0]?.value ?? "",
										})
									}
									items={datasetForSft}>
									<SelectTrigger>
										<SelectValue placeholder="Select dataset" />
									</SelectTrigger>
									<SelectContent>
										{/* <SelectItem value="defnex-defense-scenarios">
                                            defnex-defense-scenarios
                                        </SelectItem> */}
										{datasetForSft.map((dataset) => (
											<SelectItem
												key={dataset.value}
												value={dataset.value}>
												{dataset.label}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							</Field>
							<Field>
								<FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
									Immutable Version
								</FieldLabel>
								<Select
									value={value.datasetVersion}
									onValueChange={(val) =>
										onChange({ datasetVersion: val ?? "" })
									}
									items={
										datasetForSft.find(
											(data) =>
												data.value === value.dataset,
										)?.versions
									}>
									<SelectTrigger>
										<SelectValue placeholder="Select version" />
									</SelectTrigger>
									<SelectContent>
										{datasetForSft
											.find(
												(data) =>
													data.value ===
													value.dataset,
											)
											?.versions.map((version) => (
												<SelectItem
													key={version.value}
													value={version.value}>
													{version.label}
												</SelectItem>
											))}
									</SelectContent>
								</Select>
							</Field>
						</div>

						{/* Kartu Informasi Schema, Samples, Validation */}
						<div className="flex flex-col gap-2 border p-3 rounded-lg bg-white">
							<div className="grid grid-cols-3 gap-2">
								<div className="p-3 rounded-md border bg-white">
									<p className="text-xs font-semibold text-muted-foreground uppercase">
										Schema
									</p>
									<span className="font-bold text-sm">
										defnex-scenario-v1
									</span>
								</div>
								<div className="p-3 rounded-md border bg-white">
									<p className="text-xs font-semibold text-muted-foreground uppercase">
										Samples
									</p>
									<span className="font-bold text-sm">
										{datasetForSft.find(
											(data) =>
												data.value === value.dataset,
										)?.sampleCount ?? "-"}
									</span>
								</div>
								<div className="p-3 rounded-md border bg-white flex flex-col justify-between">
									<p className="text-xs font-semibold text-muted-foreground uppercase">
										Validation
									</p>
									<div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-emerald-200 bg-emerald-50 text-emerald-700 w-fit text-xs font-medium">
										<CircleWavyCheckIcon className="h-3.5 w-3.5" />
										Pass
									</div>
								</div>
							</div>

							{/* Status Immutable & Badge Ready */}
							<div className="flex justify-between items-center pt-2 px-1 text-xs text-muted-foreground">
								<div className="flex items-center gap-1.5">
									<LockIcon className="h-3.5 w-3.5" />
									<span>
										Immutable — SHA-256
										{datasetForSft.find(
											(data) =>
												data.value === value.dataset,
										)?.shaImmutable ??
											"6d2258a41920da78..."}
									</span>
								</div>
								<div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-emerald-200 bg-emerald-50 text-emerald-700 font-medium">
									<CircleWavyCheckIcon className="h-3.5 w-3.5" />
									Ready
								</div>
							</div>
						</div>

						{/* Bagian Dataset Preview */}
						<div className="flex flex-col gap-2">
							<p className="text-xs font-bold text-muted-foreground uppercase">
								Dataset Preview
							</p>
							<div className="rounded-lg bg-[#0f172a] text-slate-200 p-4 font-mono text-xs overflow-x-auto shadow-inner">
								<pre>{`{
  "scenario_id": "SCN-DEF-2026-101",
  "domain": "coastal_surveillance",
  "situation": "Uncorrelated radar return at 18 knots heading 210 with AIS beacon extinguished.",
  "recommended_actions": [
    "Dispatch FPB KRI-628 for intercept"
  ]
}`}</pre>
							</div>
						</div>
					</div>
				</FieldGroup>
			</ScrollArea>

			<div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={toggleModal}>
					Cancel
				</Button>
				<Button
					type="button"
					onClick={() => setStepActive(2)}>
					Next
				</Button>
			</div>
		</>
	);
}