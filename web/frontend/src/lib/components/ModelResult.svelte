<script lang="ts">
  import { ThumbsDown, ThumbsUp } from "lucide-svelte";
  import Button from "./ui/button/button.svelte";

  export let name: string; // name of model
  export let value: number; // likelihood of news being true
  export let description: string; // description of the model
  export let accuracy: number | undefined; // training accuracy of the model

  // Determine the color based on the value
  let color =
    value < 33
      ? "text-red-500"
      : value < 66
        ? "text-yellow-500"
        : "text-green-500";

  // Prediction messages based on likelihood of news being true
  let predictionMessage;
  value < 15
    ? (predictionMessage = "Very unlikely to be true")
    : value < 35
      ? (predictionMessage = "Unlikely to be true")
      : value < 50
        ? (predictionMessage = "Possibly true")
        : value < 75
          ? (predictionMessage = "Likely to be true")
          : value < 90
            ? (predictionMessage = "Very likely to be true")
            : (predictionMessage = "Almost certainly true");

  $: confidence = value.toFixed(2);

  async function sendFeedback(feedback: boolean) {
    const url = `http://localhost:8080/predict`;
    console.log("Feedback sent");

    try {
      fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: name,
          content: description,
          feedback,
        }),
      });
      vote = feedback;
    } catch (error) {
      console.error(error);
    }
  }
  let vote: boolean;
</script>

<!-- Enhanced styling and layout -->
<div
  class="bg-card border border-border rounded-lg p-4 mt-4 group cursor-pointer hover:shadow-lg transition-shadow duration-300"
>
  <div class="flex items-center justify-between">
    <span class="text-sm flex-1 font-cubano capitalize font-medium">{name}</span
    >
    <span class="text-sm font-medium text-muted-foreground"
      >{predictionMessage}</span
    >
    <div class="flex-1 justify-end flex gap-2 items-center">
      <span class={"text-sm  font-cubano text-right font-medium " + color}
        >{confidence}% Probability</span
      >

      {#if accuracy === undefined}
        <!-- feedback section -->
        <div class="hidden group-hover:flex items-center gap-2">
          <Button
            class="text-xs p-0 h-fit"
            on:click={() => sendFeedback(false)}
            variant="secondary"
            size="sm"
          >
            <ThumbsDown
              class={vote === false
                ? "size-3 text-red-500"
                : "size-3 text-muted-foreground hover:text-red-500"}
            />
          </Button>
          <Button
            class="text-xs p-0 h-fit"
            on:click={() => sendFeedback(true)}
            variant="secondary"
            size="sm"
          >
            <ThumbsUp
              class={vote === true
                ? "size-3 text-green-500"
                : "size-3 text-muted-foreground hover:text-green-500"}
            />
          </Button>
        </div>
      {/if}
    </div>
  </div>
  <div
    class="hidden group-hover:block transition-[height] duration-1000 ease-in-out mt-2"
  >
    <p
      class=" text-xs group-hover:h-auto transition-[height] duration-1000 ease-in-out"
    >
      {description}
      {#if accuracy !== undefined}
        <div class="flex w-full justify-end">
          <span
            class="px-2 py-1 rounded-md bg-muted border-border mt-2 text-xs text-muted-foreground"
          >
            Training accuracy: <span class="font-cubano">{accuracy}%</span>
          </span>
        </div>
      {/if}
    </p>
  </div>
</div>
