import React, { useState } from 'react';
import { Youtube, FileText, BookOpen, Brain, CheckCircle, Loader2 } from 'lucide-react';

interface SummaryResponse {
  original_text: string;
  summary: string;
  video_title: string;
}

interface QuizQuestion {
  question_text: string;
  answer?: string;
  options?: Array<{
    option: string;
    is_correct: boolean;
  }>;
}

interface QuizResponse {
  video_title: string;
  fill_in_the_blank_questions?: QuizQuestion[];
  multiple_choice_questions?: QuizQuestion[];
}

type QuizType = 'fill_in_the_blank' | 'multiple_choice' | 'both';
type Difficulty = 'easy' | 'medium' | 'hard';

function App() {
  const [url, setUrl] = useState('');
  const [summaryLength, setSummaryLength] = useState(150);
  const [loading, setLoading] = useState(false);
  const [generateQuiz, setGenerateQuiz] = useState(false);
  const [generatePdf, setGeneratePdf] = useState(false);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [quiz, setQuiz] = useState<QuizResponse | null>(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'summary' | 'quiz'>('summary');
  
  // New quiz configuration states
  const [quizType, setQuizType] = useState<QuizType>('both');
  const [numQuestions, setNumQuestions] = useState(5);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setSummary(null);
    setQuiz(null);

    try {
      // Get summary
      const summaryResponse = await fetch('http://127.0.0.1:8000/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          youtube_url: url,
          max_length: summaryLength,
          min_length: Math.min(50, summaryLength - 50)
        })
      });

      if (!summaryResponse.ok) {
        throw new Error('Failed to generate summary');
      }

      const summaryData = await summaryResponse.json();
      setSummary(summaryData);

      // Get quiz if requested
      if (generateQuiz) {
        const quizResponse = await fetch('http://127.0.0.1:8000/generate-quiz', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            youtube_url: url,
            quiz_type: quizType,
            num_questions: numQuestions,
            difficulty: difficulty
          })
        });

        if (!quizResponse.ok) {
          throw new Error('Failed to generate quiz');
        }

        const quizData = await quizResponse.json();
        setQuiz(quizData);
      }

      // Generate PDF if requested
      if (generatePdf) {
        const pdfResponse = await fetch('http://127.0.0.1:8000/generate-pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            youtube_url: url,
            pdf_type: generateQuiz ? 'summary_and_quiz' : 'summary_only',
            max_summary_length: summaryLength,
            min_summary_length: Math.min(50, summaryLength - 50),
            quiz_type: quizType,
            num_questions: numQuestions,
            difficulty: difficulty
          })
        });

        if (!pdfResponse.ok) {
          throw new Error('Failed to generate PDF');
        }

        const blob = await pdfResponse.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = 'summary.pdf';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(downloadUrl);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      {/* Hero Section */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl md:text-6xl">
              VidSummarizer
            </h1>
            <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
              Transform YouTube videos into concise summaries, interactive quizzes, and downloadable PDFs.
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="url" className="block text-sm font-medium text-gray-700">
                YouTube URL
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Youtube className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="url"
                  id="url"
                  required
                  className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-md"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label htmlFor="length" className="block text-sm font-medium text-gray-700">
                Summary Length (words)
              </label>
              <input
                type="range"
                id="length"
                min="50"
                max="300"
                value={summaryLength}
                onChange={(e) => setSummaryLength(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm text-gray-500">{summaryLength} words</span>
            </div>

            <div className="space-y-4">
              <div className="flex items-center space-x-6">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={generateQuiz}
                    onChange={(e) => setGenerateQuiz(e.target.checked)}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700">Generate Quiz</span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={generatePdf}
                    onChange={(e) => setGeneratePdf(e.target.checked)}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700">Generate PDF</span>
                </label>
              </div>

              {/* Quiz Configuration Options */}
              {generateQuiz && (
                <div className="bg-gray-50 p-4 rounded-lg space-y-4">
                  <h4 className="text-sm font-medium text-gray-900">Quiz Configuration</h4>
                  
                  <div>
                    <label htmlFor="quizType" className="block text-sm font-medium text-gray-700">
                      Quiz Type
                    </label>
                    <select
                      id="quizType"
                      value={quizType}
                      onChange={(e) => setQuizType(e.target.value as QuizType)}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    >
                      <option value="both">Both Types</option>
                      <option value="fill_in_the_blank">Fill in the Blank</option>
                      <option value="multiple_choice">Multiple Choice</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="numQuestions" className="block text-sm font-medium text-gray-700">
                      Number of Questions
                    </label>
                    <input
                      type="number"
                      id="numQuestions"
                      min="1"
                      max="10"
                      value={numQuestions}
                      onChange={(e) => setNumQuestions(Number(e.target.value))}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label htmlFor="difficulty" className="block text-sm font-medium text-gray-700">
                      Difficulty Level
                    </label>
                    <select
                      id="difficulty"
                      value={difficulty}
                      onChange={(e) => setDifficulty(e.target.value as Difficulty)}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    >
                      <option value="easy">Easy</option>
                      <option value="medium">Medium</option>
                      <option value="hard">Hard</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
                  Processing...
                </>
              ) : (
                'Generate'
              )}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {(summary || quiz) && (
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Tabs */}
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex" aria-label="Tabs">
                <button
                  onClick={() => setActiveTab('summary')}
                  className={`${
                    activeTab === 'summary'
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm`}
                >
                  Summary
                </button>
                <button
                  onClick={() => setActiveTab('quiz')}
                  className={`${
                    activeTab === 'quiz'
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm`}
                  disabled={!quiz}
                >
                  Quiz
                </button>
              </nav>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              {activeTab === 'summary' && summary && (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold text-gray-900">{summary.video_title}</h2>
                  <div className="prose max-w-none">
                    <h3 className="text-lg font-medium text-gray-900">Summary</h3>
                    <p className="mt-2 text-gray-700">{summary.summary}</p>
                  </div>
                </div>
              )}

              {activeTab === 'quiz' && quiz && (
                <div className="space-y-8">
                  <h2 className="text-2xl font-bold text-gray-900">Quiz Questions</h2>
                  
                  {quiz.fill_in_the_blank_questions && (
                    <div className="space-y-6">
                      <h3 className="text-lg font-medium text-gray-900">Fill in the Blank</h3>
                      {quiz.fill_in_the_blank_questions.map((q, i) => (
                        <div key={i} className="bg-gray-50 p-4 rounded-lg">
                          <p className="text-gray-900">{i + 1}. {q.question_text}</p>
                          <p className="mt-2 text-indigo-600 font-medium">Answer: {q.answer}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {quiz.multiple_choice_questions && (
                    <div className="space-y-6">
                      <h3 className="text-lg font-medium text-gray-900">Multiple Choice</h3>
                      {quiz.multiple_choice_questions.map((q, i) => (
                        <div key={i} className="bg-gray-50 p-4 rounded-lg">
                          <p className="text-gray-900">{i + 1}. {q.question_text}</p>
                          <div className="mt-2 space-y-2">
                            {q.options?.map((option, j) => (
                              <div key={j} className={`flex items-center ${option.is_correct ? 'text-green-600' : 'text-gray-700'}`}>
                                <CheckCircle className={`h-5 w-5 mr-2 ${option.is_correct ? 'opacity-100' : 'opacity-0'}`} />
                                <span>{option.option}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Features Section */}
      <div className="bg-white">
        <div className="max-w-7xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <div className="text-center">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                <BookOpen className="h-6 w-6" />
              </div>
              <h3 className="mt-4 text-lg font-medium text-gray-900">Smart Summaries</h3>
              <p className="mt-2 text-base text-gray-500">
                Get concise, accurate summaries of any YouTube video using advanced AI technology.
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                <Brain className="h-6 w-6" />
              </div>
              <h3 className="mt-4 text-lg font-medium text-gray-900">Interactive Quizzes</h3>
              <p className="mt-2 text-base text-gray-500">
                Generate custom quizzes to test understanding and reinforce learning.
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                <FileText className="h-6 w-6" />
              </div>
              <h3 className="mt-4 text-lg font-medium text-gray-900">PDF Export</h3>
              <p className="mt-2 text-base text-gray-500">
                Download comprehensive PDF reports with summaries and quizzes for offline use.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;