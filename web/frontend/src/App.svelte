<script lang="ts">
  import ModelResult from "$lib/components/ModelResult.svelte";
  import Navbar from "$lib/components/Navbar.svelte";
  import Button from "$lib/components/ui/button/button.svelte";
  import Input from "$lib/components/ui/input/input.svelte";
  import Label from "$lib/components/ui/label/label.svelte";
  import Textarea from "$lib/components/ui/textarea/textarea.svelte";
  import { LoaderCircle } from "lucide-svelte";
  import { ModeWatcher } from "mode-watcher";

  let title = "";
  let text = "";

  interface Prediction {
    name: string;
    value: number;
    description: string;
    accuracy: number;
  }

  let predictions: Prediction[] = [];

  $: overallPrediction = Number(
    (
      predictions.reduce((acc, curr) => acc + curr.value, 0) /
      predictions.length
    ).toFixed(2)
  );

  async function submit() {
    loading = true;

    if (title.trim().length < 4 || text.trim().length < 4) {
      loading = false;
      alert("Please fill in both field with at least 4 characters");
      return;
    }

    const url = `/predict?title=${title}&content=${text}`;

    try {
      const res = await fetch(url);

      if (res.ok) {
        const data = await res.json();
        predictions = data;
      }
    } catch (error) {
      console.error(error);
    }

    loading = false;
  }

  let loading = false;
</script>

<div class="relative flex min-h-screen flex-col bg-background">
  <Navbar />
  <div class="flex-1">
    <div class="container grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="flex flex-col py-6 gap-4">
        <h1 class="text-3xl font-bold">News Prediction</h1>
        <p class="text-lg text-muted-foreground">
          Submit a news article to get the predictions from our algorithms
        </p>

        <div class="">
          <Label class="px-3 text-lg font-semibold" for="newsTitle"
            >News Title</Label
          >
          <Input
            id="newsTitle"
            bind:value={title}
            placeholder="Enter the news title here..."
          />
        </div>
        <div class="">
          <Label class="px-3 text-lg font-semibold" for="newsText"
            >News Text</Label
          >
          <Textarea
            id="newsText"
            bind:value={text}
            class="h-48 placeholder:text-muted-foreground"
            placeholder="Enter the news text here..."
          />
        </div>

        <Button on:click={submit} disabled={loading} class="w-1/4 self-center">
          {#if loading}
            <LoaderCircle class="size-6 animate-spin" />
          {:else}
            Submit
          {/if}
        </Button>
      </div>
      <div class="flex flex-col gap-4 mb-5">
        <!-- Thhis will show the predictions for each algorithm in a simple card -->
        {#if loading || predictions.length === 0}
          <div class="flex justify-center items-center h-full">
            {#if loading}
              <LoaderCircle class="size-12 animate-spin self-center" />
            {:else}
              <p class="text-lg text-center">
                Fill in the form to get predictions for the news
              </p>
            {/if}
          </div>
        {:else}
          <div class="flex flex-col mt-4">
            <h2 class="text-xl font-bold">Overall Prediction</h2>
            <div class="flex flex-col gap-2">
              <ModelResult
                name="Overall"
                value={overallPrediction}
                accuracy={undefined}
                description="This is the overall prediction of the news, based on the predictions from all the algorithms. The percentages represent the probability of the news being real."
              />
            </div>
            <h2 class="text-xl font-bold mt-4">Predictions</h2>
            <div class="flex flex-col gap-2">
              {#each predictions as prediction}
                <ModelResult
                  name={prediction.name}
                  value={prediction.value}
                  description={prediction.description}
                  accuracy={prediction.accuracy}
                />
              {/each}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
<ModeWatcher />
