import { useState } from "react";

import DetailTrainRun from "~/components/detail-train-run";
import ModalRunSFT from "~/components/modal/run-sft";
import { columnsTrainRun } from "~/components/table/columns/train-column";
import { DataTable } from "~/components/table/data-table";
import { Button } from "~/components/ui/button";
import Header from "~/components/ui/header";

import { trainingRunsData } from "~/utils";

import { RefreshCcw } from "lucide-react";

export default function TrainRun() {
    const [modalSft, setModalSft] = useState(false);
    const [selectedRunId, setSelectedRunId] = useState<string>(trainingRunsData[0]?.runId || "");

    const toggleModalSft = () => {
        setModalSft((prevState) => !prevState);
    }

    return (
        <>
            <Header title="Training and Run" sft modalSft={toggleModalSft} />
            <div className="pl-4 pr-4 pb-4 h-full max-w-380 w-full mx-auto">
                <div className="flex justify-between items-center mb-4">
                    <div>
                        <h3 className="text-lg font-semibold">SFT Execution & Training Runs</h3>
                        <p className="text-sm text-accent-foreground">Unsloth-backed Supervised Fine-Tuning jobs and mock worker tracking</p>
                    </div>
                    <div>
                        <Button variant="outline">
                            <RefreshCcw className="mr-2 h-4 w-4" />
                            Refresh worker
                        </Button>
                    </div>
                </div>

                <div className="grid xl:grid-cols-3 lg:grid-cols-4 grid-cols-1 gap-4">
                    <div className="flex flex-col col-span-2 bg-white border rounded-md p-4 gap-4">
                        <DataTable columns={columnsTrainRun(setSelectedRunId)} data={trainingRunsData} trainRun />
                    </div>
                    <div className="xl:col-span-1 lg:col-span-2 col-span-1 flex flex-col gap-4">
                        {trainingRunsData.map((run) => {
                            if (run.runId === selectedRunId) {
                                return (
                                    <DetailTrainRun run={run} />
                                )
                            }
                        })}
                    </div>
                </div>
            </div>

            {/* {modalSft && (
            )} */}
            <ModalRunSFT isOpen={modalSft} toggleModal={toggleModalSft} />
        </>
    )
}