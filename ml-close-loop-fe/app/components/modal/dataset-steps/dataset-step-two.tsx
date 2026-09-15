import {
	Field,
	FieldDescription,
	FieldGroup,
	FieldLabel,
} from "../../ui/field";
import { Label } from "../../ui/label";
import { Input } from "../../ui/input";
import { CircleCheck } from "lucide-react";
import clsx from "clsx";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "../../ui/select";
import { Textarea } from "~/components/ui/textarea";
import { useEffect, useState } from "react";
import { Button } from "~/components/ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";

export type StepTwoValueProps = {
	versionMode: "add" | "create";
	datasetId: string;
	datasetName: string;
	datasetVersion: string;
	intakeNotes?: string;
};

export type DatasetVerionProps = {
	name: string;
	version: string;
};

export function DatasetStepTwo({
	value,
	onChange,
	setStepActive,
}: {
	value: StepTwoValueProps;
	onChange: (patch: Partial<StepTwoValueProps>) => void;
	setStepActive: (step: number) => void;
}) {
	const [versionMode, setVersionMode] = useState<"add" | "create">("add");
	const [datasetVersion, setDatasetVersion] = useState<DatasetVerionProps[]>(
		[],
	);
	const [targetVersion, setTargetVersion] = useState<string>("v1.0.0");
	const [suggestedNextVersion, setSuggestedNextVersion] =
		useState<string>("v1.0.0");

	const methods = [
		{
			label: `Add Version to Existing Dataset (${datasetVersion.length})`,
			value: "add",
		},
		{ label: "Create New Dataset Catalog", value: "create" },
	] as const;

	const handleChangeMethod = () => {
		if (datasetVersion.length <= 0) {
			setVersionMode("create");
		} else {
			if (versionMode === "add") {
				setVersionMode("create");
			} else {
				setVersionMode("add");
			}
		}
	};

	useEffect(() => {
		function checkDatasetVersion() {
			if (datasetVersion.length === 0) {
				setVersionMode("create");
			} else {
				setVersionMode("add");
				const targetVersion =
					datasetVersion[datasetVersion.length - 1].version;
				setTargetVersion(targetVersion);
				const versionParts = targetVersion.split(".");
				const nextVersion = `${versionParts[0]}.${versionParts[1]}.${
					parseInt(versionParts[2]) + 1
				}`;
				setSuggestedNextVersion(nextVersion);
			}
		}
		checkDatasetVersion();
	}, []);

	const isAvailabletoNextStep = () => {
		if (versionMode === "add") {
			return value.datasetId && value.datasetVersion;
		} else {
			return value.datasetId && value.datasetName && value.datasetVersion;
		}
	};

	return (
		<>
			<ScrollArea className="max-h-125">
				<FieldGroup>
					<h3>Select Dataset Source Type</h3>
					<div className="grid grid-cols-2 gap-4">
						{methods.map((method) => (
							<Field
								key={method.value}
								className={clsx(
									"flex flex-col gap-1 p-2 rounded-lg border",
									versionMode === method.value
										? "border-black"
										: "border-gray-300 text-gray-300",
									datasetVersion.length <= 0 &&
										method.value === "add"
										? "opacity-50 cursor-not-allowed"
										: "cursor-pointer",
								)}
								onClick={handleChangeMethod}>
								<p>{method.label}</p>
							</Field>
						))}

						{versionMode === "add" ? (
							<div className="flex flex-col gap-2 rounded-lg w-full col-span-2">
								<Field>
									<Label>Choose Dataset</Label>
									<Select
										items={datasetVersion.map((f) => ({
											label: f.name,
											value: f.name + f.version,
										}))}
										defaultValue={
											datasetVersion[0]?.name +
											datasetVersion[0]?.version
										}
										onValueChange={(e) => {
											onChange({
												datasetId:
													e?.split(
														/(?<=\D)(?=\d)/,
													)[0],
												datasetVersion:
													e?.split(
														/(?<=\D)(?=\d)/,
													)[1],
											});
											console.log(
												"Selected Dataset:",
												value.datasetId,
												value.datasetVersion,
											);
										}}>
										<SelectTrigger>
											<SelectValue placeholder="Select Dataset" />
										</SelectTrigger>
										<SelectContent>
											{datasetVersion.map((item) => (
												<SelectItem
													key={item.name}
													value={
														item.name + item.version
													}>
													{item.name} - {item.version}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								</Field>
								<div className="flex flex-col p-2 bg-gray-600/10 rounded-md">
									<div className="flex justify-between">
										<p>Current latest version:</p>
										<span>
											{value.datasetVersion ?? "-"}
										</span>
									</div>
									<div className="flex justify-between">
										<p>Suggested next increment:</p>
										<span>{suggestedNextVersion}</span>
									</div>
								</div>

								<Field>
									<Label>
										Target Version Tag{" "}
										<span className="text-red-500">*</span>
									</Label>
									<div className="relative">
										<Input
											placeholder="e.g. v1.2.0"
											className="h-10"
											value={
												value.datasetVersion ??
												targetVersion
											}
											onChange={(e) =>
												onChange({
													datasetVersion:
														e.target.value,
												})
											}
										/>
										<div className="absolute right-3 top-2.5 flex gap-1 items-center text-green-400 border border-green-500 bg-green-50 px-1 rounded-sm font-semibold">
											<CircleCheck className="text-green h-4 w-4" />
											<span>Imutable</span>
										</div>
									</div>
									<FieldDescription>
										Dataset versions are immutable. Once
										created, version v1.2.0 cannot be
										modified or replaced.
									</FieldDescription>
								</Field>
							</div>
						) : (
							<div className="grid md:grid-cols-2 grid-cols-1 gap-2 p-2 rounded-lg w-full col-span-2">
								<Field>
									<Label>
										New Dataset ID{" "}
										<span className="text-red-500">*</span>
									</Label>
									<Input
										placeholder="e.g. ds-defense-screnarios"
										onChange={(e) => {
											const formattedValue =
												e.target.value
													.toLowerCase() // 1. Ubah semua teks menjadi huruf kecil
													.replace(/\s+/g, "-");
											onChange({
												datasetId: formattedValue,
											});
										}}
										value={value.datasetId ?? ""}
									/>
								</Field>
								<Field>
									<Label>
										Dataset Display Name{" "}
										<span className="text-red-500">*</span>
									</Label>
									<Input
										placeholder="e.g. Defense Scenarios Corpus"
										onChange={(e) => {
											onChange({
												datasetName: e.target.value,
											});
										}}
										value={value.datasetName ?? ""}
									/>
								</Field>
								<Field className="md:col-span-2">
									<Label>
										Target Version Tag{" "}
										<span className="text-red-500">*</span>
									</Label>
									<div className="relative">
										<Input
											placeholder="e.g. v1.0.0"
											type="text"
											className="h-10"
											value={
												value.datasetVersion ?? "v1.0.0"
											}
											onChange={(e) =>
												onChange({
													datasetVersion:
														e.target.value,
												})
											}
										/>
										<div className="absolute right-3 top-2.5 flex gap-1 items-center text-green-400 border border-green-500 bg-green-50 px-1 rounded-sm font-semibold">
											<CircleCheck className="text-green h-4 w-4" />
											<span>Imutable</span>
										</div>
									</div>
									<FieldDescription>
										Dataset versions are immutable. Once
										created, version v1.2.0 cannot be
										modified or replaced.
									</FieldDescription>
								</Field>
							</div>
						)}

						{/* footer */}
						<div className="col-span-2">
							<div className="flex flex-col gap-4 p-2 rounded-lg">
								<Field>
									<Label>
										Ingested By (Authenticated Operator)
									</Label>
									<div className="flex flex-wrap justify-between items-center gap-2 bg-gray-600/10 p-2 rounded-md">
										<div className="flex items-center gap-1 text-xs">
											<div className="rounded-full bg-amber-200 p-1 w-8 h-8 flex justify-center items-center">
												M
											</div>
											<p>Maulana M.</p>
											<span>(ML_ENGINEER)</span>
										</div>
										<p className="text-slate-500 font-semibold">
											READ-ONLY SYSTEM CONTEXT
										</p>
									</div>
								</Field>
								<Field>
									<Label htmlFor="intake-notes">
										Intake Notes / Description (Optional)
									</Label>
									<Textarea
										id="intake-notes"
										placeholder="Enter intake notes..."
										onChange={(e) =>
											onChange({
												intakeNotes: e.target.value,
											})
										}
										value={value.intakeNotes ?? ""}
									/>
								</Field>
							</div>
						</div>
					</div>
				</FieldGroup>
			</ScrollArea>
			<div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(1)}>
					Back
				</Button>
				<Button
					type="button"
					disabled={!isAvailabletoNextStep()}
					className={`${isAvailabletoNextStep() ? "" : "opacity-50 cursor-not-allowed"}`}
					onClick={() => setStepActive(3)}>
					Next
				</Button>
			</div>
		</>
	);
}
