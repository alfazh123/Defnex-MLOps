import { Button } from "../ui/button";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog"
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "../ui/dialog";
import { useState } from "react";
import { clsx } from "clsx";
import {
	SftStepOne,
	type StepOneValueProps,
} from "./rus-sft-steps/sft-step-one";
import {
	SftStepTwo,
	type stepTwoValueProps,
} from "./rus-sft-steps/sft-step-two";
import {
	SftStepThree,
	type StepThreeValueProps,
} from "./rus-sft-steps/sft-step-three";
import {
	SftStepFour,
	type StepFourValueProps,
} from "./rus-sft-steps/sft-step-four";
import {
	SftStepFive,
	type StepFiveValueProps,
} from "./rus-sft-steps/sft-step-five";
import {
	SftStepSix,
	type StepSixValueProps,
} from "./rus-sft-steps/sft-step-six";
import {
	SftStepResponse,
	type StepResponseValueProps,
} from "./rus-sft-steps/sft-step-response";
import { baseModelsSftModal, datasetForSft, providers } from "~/utils";
import { CheckIcon, XIcon } from "@phosphor-icons/react/dist/ssr";

const steps = [
	{ id: 1, name: "Dataset" },
	{ id: 2, name: "Model" },
	{ id: 3, name: "Compute" },
	{ id: 4, name: "Training Config" },
	{ id: 5, name: "Evaluation" },
	{ id: 6, name: "Review" },
];

export type LoraConfigPropsSft = {
	rank: number;
	alpha: number;
	dropout: number;
	targetModules: string[];
};

export type TrainingParapPropsSft = {
	epochs: number;
	learningRate: number;
	batchSize: number;
	gradientAccumulationSteps: number;
	warmupRatio: number;
	weightDecay: number;
	maxSeqLength: number;
	optimizer: string;
	scheduler: string;
};

export default function ModalRunSFT({ isOpen, toggleModal }: { isOpen: boolean, toggleModal: () => void }) {
    const [stepActive, setStepActive] = useState(1);

	// step 1 state
	const [dataset, setDataset] = useState<string>(datasetForSft[0].value);
	const [datasetVersion, setDatasetVersion] = useState<string>(
		datasetForSft[0].versions[0].value,
	);

	// step 2 state
	const [baseModel, setBaseModel] = useState<string>(
		baseModelsSftModal[0].name,
	);
	const [baseModelVersion, setBaseModelVersion] = useState<string>(
		baseModelsSftModal[0].version,
	);

	// step 3 state
	const [provider, setProvider] = useState<string>(providers[0].name);
	const [computeRes, setComputeRes] = useState<string>(
		providers[0].computeRes[0].name,
	);

	// step 4 state
	const [loraConf, setLoraConf] = useState<LoraConfigPropsSft>({
		rank: 16,
		alpha: 32,
		dropout: 0.05,
		targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"],
	});
	const [trainingParams, setTrainingParams] = useState<TrainingParapPropsSft>(
		{
			epochs: 3,
			learningRate: 0.0002,
			batchSize: 2,
			gradientAccumulationSteps: 1,
			warmupRatio: 0.03,
			weightDecay: 0.01,
			maxSeqLength: 512,
			optimizer: "AdamW",
			scheduler: "linear",
		},
	);

	// step 5 state
	const [goldenSet, setGoldenSet] = useState<string>(
		"defnex-defense-scenarios-eval",
	);
	const [qaGate, setQaGate] = useState({
		structuralValidity: 95,
		domainQuality: 80,
	});

	// response run sft modal
	const [runSftId, setRunSftId] = useState<string>("");

	// form values mapping
	const stepOneValue: StepOneValueProps = {
		dataset,
		datasetVersion,
	};

	const stepTwoValue: stepTwoValueProps = {
		baseModel,
		baseModelVersion,
	};

	const stepThreeValue: StepThreeValueProps = {
		provider,
		computeRes,
	};

	const stepFourValue: StepFourValueProps = {
		loraConf,
		trainingParams,
	};

	const stepFiveValue: StepFiveValueProps = {
		goldenSet,
		qaGate,
	};

	const stepSixValue: StepSixValueProps = {
		dataset: dataset + " @ " + datasetVersion,
		baseModel: baseModel + " @ " + baseModelVersion,
		compute: provider + " • " + computeRes,
		training:
			trainingParams.epochs +
			" epochs • lr " +
			trainingParams.learningRate +
			" • batch " +
			trainingParams.batchSize,
		samplesValidation: goldenSet + " • PASS",
		resultingModelVersion: baseModel + "-" + baseModelVersion,
		lora:
			"r=" +
			loraConf.rank +
			" α=" +
			loraConf.alpha +
			" dropout=" +
			loraConf.dropout,
		evaluation: goldenSet + " • Gate Configured",
	};

	const stepResponseValue: StepResponseValueProps = {
		runSftId,
	};

	// Update functions for each step (mengikuti pola file pertama)
	const updateStepOne = (patch: Partial<StepOneValueProps>) => {
		if ("dataset" in patch) setDataset(patch.dataset ?? "");
		if ("datasetVersion" in patch)
			setDatasetVersion(patch.datasetVersion ?? "");
	};

	const updateStepTwo = (patch: Partial<stepTwoValueProps>) => {
		if ("baseModel" in patch) setBaseModel(patch.baseModel ?? "");
		if ("baseModelVersion" in patch)
			setBaseModelVersion(patch.baseModelVersion ?? "");
	};

	const updateStepThree = (patch: Partial<StepThreeValueProps>) => {
		if ("provider" in patch) setProvider(patch.provider ?? "");
		if ("computeResource" in patch) setComputeRes(patch.computeRes ?? "");
	};

	const updateStepFour = (patch: Partial<StepFourValueProps>) => {
		if ("loraConf" in patch && patch.loraConf) setLoraConf(patch.loraConf);
		if ("trainingParams" in patch && patch.trainingParams)
			setTrainingParams(patch.trainingParams);
	};

	const updateStepFive = (patch: Partial<StepFiveValueProps>) => {
		if ("goldenSet" in patch) setGoldenSet(patch.goldenSet ?? "");
		if ("qaGate" in patch && patch.qaGate) setQaGate(patch.qaGate);
	};

	const resetForm = () => {
		setStepActive(1);
		setDataset("");
		setDatasetVersion("");
		setBaseModel("");
		setBaseModelVersion("");
		setProvider("");
		setComputeRes("");
		setGoldenSet("");
	};

    return (
		<Dialog open={isOpen}>
			<form>
				<DialogContent
					className="sm:max-w-4xl w-full"
					showCloseButton={false}>
					<DialogHeader>
						<DialogTitle>Launch SFT Training Run</DialogTitle>
						<DialogDescription>
							Queue an SFT fine-tuning job with Unsloth
							configuration
						</DialogDescription>
					</DialogHeader>
					{stepActive <= steps.length && (
						<div className="grid grid-cols-6">
							{steps.map((step) => (
								<div
									key={step.id}
									className="flex flex-col justify-center items-center gap-1 p-2 rounded-lg">
									<div
										className={clsx(
											"w-10 h-10 flex items-center justify-center",
											stepActive === step.id
												? "bg-black text-white rounded-full"
												: stepActive > step.id
													? "bg-black text-white rounded-full"
													: "bg-gray-200 text-gray-500 rounded-full",
										)}>
										{stepActive > step.id ? (
											<CheckIcon className="w-5 h-5" />
										) : (
											<span className="font-semibold">
												{step.id}
											</span>
										)}
									</div>
									<span className="font-semibold sm:flex hidden">
										{step.name}
									</span>
								</div>
							))}
						</div>
					)}

					{stepActive === 1 && (
						<SftStepOne
							value={stepOneValue}
							onChange={updateStepOne}
							setStepActive={setStepActive}
							toggleModal={toggleModal}
						/>
					)}
					{stepActive === 2 && (
						<SftStepTwo
							value={stepTwoValue}
							onChange={updateStepTwo}
							setStepActive={setStepActive}
						/>
					)}
					{stepActive === 3 && (
						<SftStepThree
							value={stepThreeValue}
							onChange={updateStepThree}
							setStepActive={setStepActive}
						/>
					)}
					{stepActive === 4 && (
						<SftStepFour
							value={stepFourValue}
							onChange={updateStepFour}
							setStepActive={setStepActive}
						/>
					)}
					{stepActive === 5 && (
						<SftStepFive
							value={stepFiveValue}
							onChange={updateStepFive}
							setStepActive={setStepActive}
						/>
					)}
					{stepActive === 6 && (
						<SftStepSix
							value={stepSixValue}
							setStepActive={setStepActive}
							// submitSft={() => {
							//     console.log("submit sft training", requestData);
							// }}
						/>
					)}
					{stepActive === 7 && (
						<SftStepResponse
							value={stepResponseValue}
							toggleModal={toggleModal}
							resetForm={resetForm}
						/>
					)}

					<DialogPrimitive.Close
						onClick={() => {
							toggleModal();
							resetForm();
						}}
						data-slot="dialog-close"
						render={
							<Button
								variant="ghost"
								className="absolute top-4 right-4 bg-secondary"
								size="icon-sm"
							/>
						}>
						<XIcon />
						<span className="sr-only">Close</span>
					</DialogPrimitive.Close>
				</DialogContent>
			</form>
		</Dialog>
	);
}