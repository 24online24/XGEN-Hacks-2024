<script lang="ts">
  import ModelResult from "$lib/components/ModelResult.svelte";
  import Navbar from "$lib/components/Navbar.svelte";
  import Button from "$lib/components/ui/button/button.svelte";
  import Input from "$lib/components/ui/input/input.svelte";
  import Label from "$lib/components/ui/label/label.svelte";
  import Textarea from "$lib/components/ui/textarea/textarea.svelte";
  import { LoaderCircle } from "lucide-svelte";
  import { ModeWatcher } from "mode-watcher";

  let title = "Hello World";
  let text = "This is a test text";

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
    console.log(title, text);
    // this will call the backend to get the predictions, each algoritm has the following object
    // {name: string, value: number, description: string}

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
  <div class="flex-1 pt-2">
    <div class="container grid grid-cols-2 gap-6">
      <div class="flex flex-col py-6 gap-4">
        <!-- header -->
        <h1 class="text-3xl font-bold">News Prediction</h1>
        <!-- Form to submit the news -->
        <!-- desc -->
        <p class="text-lg">
          Submit a news article to get the predictions from our algorithms
        </p>

        <Label class="px-3" for="newsTitle">News Title</Label>
        <Input
          id="newsTitle"
          bind:value={title}
          placeholder="Enter the news text here..."
        />
        <Label class="px-3" for="newsText">News Text</Label>
        <Textarea
          id="newsText"
          bind:value={text}
          class="h-48"
          placeholder="Enter the news text here..."
        />
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
              <p class="text-lg text-center">No predictions available</p>
            {/if}
          </div>
        {:else}
          <div class="flex flex-col mt-4">
            <h2 class="text-xl font-bold">Overall Prediction</h2>
            <div class="flex flex-col gap-2">
              <ModelResult
                name="Overall"
                value={overallPrediction}
                description="This is the overall prediction of the news"
                accuracy={0}
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
