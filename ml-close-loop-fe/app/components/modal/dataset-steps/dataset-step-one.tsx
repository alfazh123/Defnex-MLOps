import {
	Field,
	FieldDescription,
	FieldGroup,
	FieldLabel,
} from "../../ui/field";
import { Label } from "../../ui/label";
import { Input } from "../../ui/input";
import { Separator } from "../../ui/separator";
import { Button } from "../../ui/button";
import clsx from "clsx";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "../../ui/select";
import { processDatasetFile } from "../../../lib/dataset-format";
import { ScrollArea } from "~/components/ui/scroll-area";
import {
	CheckCircleIcon,
	FileTextIcon,
	PackageIcon,
	RepeatIcon,
	UploadIcon,
} from "@phosphor-icons/react/dist/ssr";

export type StepOneValueProps = {
	sourceMethod: "file" | "hf";
	sourceFile: File | null;
	hfRepoId: string;
	split: string;
	revision: string;
	detectedFormat: string;
	detectedSample: number;
};

const methods = [
	{
		label: "Upload File",
		description: "Local JSONL, CSV, or Excel files parsed & normalized",
		value: "file",
	},
	{
		label: "Hugging Face Dataset",
		description:
			"Direct import from public or private Hugging Face hub repository",
		value: "hf",
	},
] as const;

export function DatasetStepOne({
	value,
	onChange,
	setStepActive,
	toggleModal,
}: {
	value: StepOneValueProps;
	onChange: (patch: Partial<StepOneValueProps>) => void;
	setStepActive: (step: number) => void;
	toggleModal: () => void;
}) {
	const handleFilePicked = (file: File | null) => {
		onChange({ sourceFile: file });
		if (!file) return;
		void processDatasetFile(file)
			.then((result) =>
				onChange({
					detectedFormat: result.format,
					detectedSample: result.sampleCount,
				}),
			)
			.catch(() =>
				onChange({
					detectedFormat: "Unknown Format",
					detectedSample: 0,
				}),
			);
	};

	const fileSize = value.sourceFile
		? `${(value.sourceFile.size / (1024 * 1024)).toFixed(2)} MB`
		: "--";

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
									"flex flex-col gap-1 p-2 rounded-lg border cursor-pointer",
									value.sourceMethod === method.value
										? "border-black"
										: "border-gray-300 text-gray-300",
								)}
								onClick={() =>
									onChange({ sourceMethod: method.value })
								}>
								<FieldLabel>{method.label}</FieldLabel>
								<FieldDescription>
									{method.description}
								</FieldDescription>
							</Field>
						))}

						{value.sourceMethod === "file" ? (
							<div className="flex flex-col gap-2 rounded-lg w-full col-span-2 relative">
								{value.sourceFile ? (
									<Label
										htmlFor="file-upload-input"
										className="flex justify-between w-full p-2 rounded-sm border-dashed border-2 border-gray-300 items-center">
										{value.sourceFile.name}
										<RepeatIcon className="h-6 w-6 text-black" />
									</Label>
								) : (
									<Label
										htmlFor="file-upload-input"
										className="flex flex-col h-52 gap-2 border-dashed w-full items-center justify-center border-2 border-gray-300 rounded-lg cursor-pointer hover:bg-gray-100">
										<div className="flex flex-col gap-1 items-center justify-center bg-gray-200 p-2 rounded-full">
											<UploadIcon className="mx-auto h-6 w-6 text-black" />
										</div>
										<p className="text-base font-semibold">
											Click to browse or drag and drop
											dataset file
										</p>
										<span className="text-xs text-muted-foreground">
											Supported: JSONL, CSV, Excel (.xlsx)
										</span>
									</Label>
								)}
								<Input
									id="file-upload-input"
									type="file"
									accept=".jsonl,.csv,.xlsx"
									className="hidden"
									onChange={(e) => {
										const file =
											e.target.files?.[0] || null;
										handleFilePicked(file);
										e.target.value = "";
									}}
								/>

								<div className="flex gap-1 p-2 rounded-lg border border-yellow-300 text-yellow-600 bg-amber-50 text-xs">
									<FileTextIcon className="h-8 w-8 text-yellow-600" />
									<p className="font-light">
										<span className="font-semibold">
											Format Normalization:
										</span>{" "}
										Non-JSONL files (such as CSV or Excel)
										will be automatically converted and
										validated into canonical JSONL ShareGPT
										format for training pipelines.
									</p>
								</div>
							</div>
						) : (
							<div className="flex flex-col gap-2 p-2 rounded-lg w-full col-span-2">
								<Field>
									<FieldLabel htmlFor="hf-dataset-input">
										Hugging Face Hub Repository ID
									</FieldLabel>
									<div className="flex gap-2 ">
										<Input
											id="hf-dataset-input"
											placeholder="e.g. username/dataset-name"
											onChange={(e) =>
												onChange({
													hfRepoId: e.target.value,
												})
											}
											value={
												value.hfRepoId
													? value.hfRepoId
													: ""
											}
										/>
										<Button>Validate</Button>
									</div>
									<FieldDescription className="text-xs">
										Examples:{" "}
										<span className="text-black">
											defnex/defense-scenarios-v1,
											Open-Orca/OpenOrca, tatsu-lab/alpaca
										</span>
									</FieldDescription>
								</Field>
								<div className="grid grid-cols-2 gap-2">
									<Field>
										<FieldLabel htmlFor="dataset-split">
											Dataset Split
										</FieldLabel>
										<Select
											id="dataset-split"
											value={value.split}
											onValueChange={(v) =>
												onChange({ split: v ?? "" })
											}>
											<SelectTrigger>
												<SelectValue placeholder="Select a split" />
											</SelectTrigger>
											<SelectContent>
												<SelectItem value="train">
													Train
												</SelectItem>
												<SelectItem value="test">
													Test
												</SelectItem>
												<SelectItem value="validation">
													Validation
												</SelectItem>
											</SelectContent>
										</Select>
									</Field>
									<Field>
										<FieldLabel htmlFor="revision">
											Revision / Git Branch
										</FieldLabel>
										<Input
											id="revision"
											placeholder="Enter revision or branch name"
											onChange={(e) =>
												onChange({
													revision: e.target.value,
												})
											}
											value={
												value.revision
													? value.revision
													: ""
											}
										/>
									</Field>
								</div>
							</div>
						)}
						{value.sourceFile || value.hfRepoId ? (
							<div className="col-span-2">
								<div className="flex flex-col gap-2 border p-2 rounded-lg bg-emerald-50/40 border-emerald-200">
									<div className="flex justify-between items-center">
										<div className="flex gap-2 items-center text-emerald-700">
											<CheckCircleIcon className="h-4 w-4" />
											<p className="text-sm font-semibold">
												{value.sourceFile?.name || ""}
											</p>
										</div>
									</div>
									<Separator />
									<div className="grid md:grid-cols-3 grid-cols-1 gap-2">
										<div className="p-2 rounded-md border bg-white">
											<p className="font-semibold text-muted-foreground">
												Sample Detected
											</p>
											<div className="flex gap-1 items-center">
												<PackageIcon className="h-4 w-4 text-accent-foreground" />
												<span className="font-bold break-all">
													{value.detectedSample ||
														"--"}
												</span>
											</div>
										</div>
										<div className="p-2 rounded-md border bg-white">
											<p className="font-semibold text-muted-foreground">
												Ukuran File
											</p>
											<span className="font-bold">
												{fileSize}
											</span>
										</div>
										<div className="p-2 rounded-md border bg-white">
											<p className="font-semibold text-muted-foreground">
												Format Normalized
											</p>
											<span className="font-bold">
												{value.detectedFormat ||
													"JSONL"}
											</span>
										</div>
									</div>
								</div>
							</div>
						) : (
							<div></div>
						)}
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
					className={`${value.sourceFile || value.hfRepoId ? "" : "opacity-50 cursor-not-allowed"}`}
					disabled={!value.sourceFile && !value.hfRepoId}
					onClick={() => setStepActive(2)}>
					Next
				</Button>
			</div>
		</>
	);
}
